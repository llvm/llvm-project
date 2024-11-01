//===-- ConvertConstant.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertConstant.h"
#include "flang/Evaluate/expression.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/Mangler.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/Todo.h"

/// Convert string, \p s, to an APFloat value. Recognize and handle Inf and
/// NaN strings as well. \p s is assumed to not contain any spaces.
static llvm::APFloat consAPFloat(const llvm::fltSemantics &fsem,
                                 llvm::StringRef s) {
  assert(!s.contains(' '));
  if (s.compare_insensitive("-inf") == 0)
    return llvm::APFloat::getInf(fsem, /*negative=*/true);
  if (s.compare_insensitive("inf") == 0 || s.compare_insensitive("+inf") == 0)
    return llvm::APFloat::getInf(fsem);
  // TODO: Add support for quiet and signaling NaNs.
  if (s.compare_insensitive("-nan") == 0)
    return llvm::APFloat::getNaN(fsem, /*negative=*/true);
  if (s.compare_insensitive("nan") == 0 || s.compare_insensitive("+nan") == 0)
    return llvm::APFloat::getNaN(fsem);
  return {fsem, s};
}

//===----------------------------------------------------------------------===//
// Fortran::lower::tryCreatingDenseGlobal implementation
//===----------------------------------------------------------------------===//

/// Generate an mlir attribute from a literal value
template <Fortran::common::TypeCategory TC, int KIND>
static mlir::Attribute convertToAttribute(
    fir::FirOpBuilder &builder,
    const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>> &value,
    mlir::Type type) {
  if constexpr (TC == Fortran::common::TypeCategory::Integer) {
    return builder.getIntegerAttr(type, value.ToInt64());
  } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
    return builder.getIntegerAttr(type, value.IsTrue());
  } else {
    static_assert(TC == Fortran::common::TypeCategory::Real,
                  "type values cannot be converted to attributes");
    std::string str = value.DumpHexadecimal();
    auto floatVal =
        consAPFloat(builder.getKindMap().getFloatSemantics(KIND), str);
    return builder.getFloatAttr(type, floatVal);
  }
  return {};
}

namespace {
/// Helper class to lower an array constant to a global with an MLIR dense
/// attribute.
///
/// If we have a rank-1 array of integer, real, or logical, then we can
/// create a global array with the dense attribute.
///
/// The mlir tensor type can only handle integer, real, or logical. It
/// does not currently support nested structures which is required for
/// complex.
///
/// Also, we currently handle just rank-1 since tensor type assumes
/// row major array ordering. We will need to reorder the dimensions
/// in the tensor type to support Fortran's column major array ordering.
/// How to create this tensor type is to be determined.
class DenseGlobalBuilder {
public:
  static fir::GlobalOp tryCreating(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Type symTy,
                                   llvm::StringRef globalName,
                                   mlir::StringAttr linkage, bool isConst,
                                   const Fortran::lower::SomeExpr &initExpr) {
    DenseGlobalBuilder globalBuilder;
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Expr<Fortran::evaluate::SomeLogical> &
                    x) { globalBuilder.tryConvertingToAttributes(builder, x); },
            [&](const Fortran::evaluate::Expr<Fortran::evaluate::SomeInteger> &
                    x) { globalBuilder.tryConvertingToAttributes(builder, x); },
            [&](const Fortran::evaluate::Expr<Fortran::evaluate::SomeReal> &x) {
              globalBuilder.tryConvertingToAttributes(builder, x);
            },
            [](const auto &) {},
        },
        initExpr.u);
    return globalBuilder.tryCreatingGlobal(builder, loc, symTy, globalName,
                                           linkage, isConst);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  static fir::GlobalOp tryCreating(
      fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type symTy,
      llvm::StringRef globalName, mlir::StringAttr linkage, bool isConst,
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &constant) {
    DenseGlobalBuilder globalBuilder;
    globalBuilder.tryConvertingToAttributes(builder, constant);
    return globalBuilder.tryCreatingGlobal(builder, loc, symTy, globalName,
                                           linkage, isConst);
  }

private:
  DenseGlobalBuilder() = default;

  /// Try converting an evaluate::Constant to a list of MLIR attributes.
  template <Fortran::common::TypeCategory TC, int KIND>
  void tryConvertingToAttributes(
      fir::FirOpBuilder &builder,
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &constant) {
    static_assert(TC != Fortran::common::TypeCategory::Character,
                  "must be numerical or logical");
    if (constant.Rank() != 1)
      return;
    auto attrTc = TC == Fortran::common::TypeCategory::Logical
                      ? Fortran::common::TypeCategory::Integer
                      : TC;
    attributeElementType = Fortran::lower::getFIRType(
        builder.getContext(), attrTc, KIND, std::nullopt);
    for (auto element : constant.values())
      attributes.push_back(
          convertToAttribute<TC, KIND>(builder, element, attributeElementType));
  }

  /// Try converting an evaluate::Expr to a list of MLIR attributes.
  template <typename SomeCat>
  void tryConvertingToAttributes(fir::FirOpBuilder &builder,
                                 const Fortran::evaluate::Expr<SomeCat> &expr) {
    std::visit(
        [&](const auto &x) {
          using TR = Fortran::evaluate::ResultType<decltype(x)>;
          if (const auto *constant =
                  std::get_if<Fortran::evaluate::Constant<TR>>(&x.u))
            tryConvertingToAttributes<TR::category, TR::kind>(builder,
                                                              *constant);
        },
        expr.u);
  }

  /// Create a fir::Global if MLIR attributes have been successfully created by
  /// tryConvertingToAttributes.
  fir::GlobalOp tryCreatingGlobal(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Type symTy,
                                  llvm::StringRef globalName,
                                  mlir::StringAttr linkage,
                                  bool isConst) const {
    // Not a rank 1 "trivial" intrinsic constant array, or empty array.
    if (!attributeElementType || attributes.empty())
      return {};

    auto tensorTy =
        mlir::RankedTensorType::get(attributes.size(), attributeElementType);
    auto init = mlir::DenseElementsAttr::get(tensorTy, attributes);
    return builder.createGlobal(loc, symTy, globalName, linkage, init, isConst);
  }

  llvm::SmallVector<mlir::Attribute> attributes;
  mlir::Type attributeElementType;
};
} // namespace

fir::GlobalOp Fortran::lower::tryCreatingDenseGlobal(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type symTy,
    llvm::StringRef globalName, mlir::StringAttr linkage, bool isConst,
    const Fortran::lower::SomeExpr &initExpr) {
  return DenseGlobalBuilder::tryCreating(builder, loc, symTy, globalName,
                                         linkage, isConst, initExpr);
}

//===----------------------------------------------------------------------===//
// Fortran::lower::IntrinsicConstantBuilder<TC, KIND>::gen
// Lower an array constant to a fir::ExtendedValue.
//===----------------------------------------------------------------------===//

/// Generate a real constant with a value `value`.
template <int KIND>
static mlir::Value genRealConstant(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   const llvm::APFloat &value) {
  mlir::Type fltTy = Fortran::lower::convertReal(builder.getContext(), KIND);
  return builder.createRealConstant(loc, fltTy, value);
}

/// Convert a scalar literal constant to IR.
template <Fortran::common::TypeCategory TC, int KIND>
static mlir::Value genScalarLit(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>> &value) {
  if constexpr (TC == Fortran::common::TypeCategory::Integer) {
    mlir::Type ty = Fortran::lower::getFIRType(builder.getContext(), TC, KIND,
                                               std::nullopt);
    if (KIND == 16) {
      auto bigInt =
          llvm::APInt(ty.getIntOrFloatBitWidth(), value.SignedDecimal(), 10);
      return builder.create<mlir::arith::ConstantOp>(
          loc, ty, mlir::IntegerAttr::get(ty, bigInt));
    }
    return builder.createIntegerConstant(loc, ty, value.ToInt64());
  } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
    return builder.createBool(loc, value.IsTrue());
  } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
    std::string str = value.DumpHexadecimal();
    if constexpr (KIND == 2) {
      auto floatVal = consAPFloat(llvm::APFloatBase::IEEEhalf(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else if constexpr (KIND == 3) {
      auto floatVal = consAPFloat(llvm::APFloatBase::BFloat(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else if constexpr (KIND == 4) {
      auto floatVal = consAPFloat(llvm::APFloatBase::IEEEsingle(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else if constexpr (KIND == 10) {
      auto floatVal = consAPFloat(llvm::APFloatBase::x87DoubleExtended(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else if constexpr (KIND == 16) {
      auto floatVal = consAPFloat(llvm::APFloatBase::IEEEquad(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else {
      // convert everything else to double
      auto floatVal = consAPFloat(llvm::APFloatBase::IEEEdouble(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    }
  } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
    mlir::Value realPart =
        genScalarLit<Fortran::common::TypeCategory::Real, KIND>(builder, loc,
                                                                value.REAL());
    mlir::Value imagPart =
        genScalarLit<Fortran::common::TypeCategory::Real, KIND>(builder, loc,
                                                                value.AIMAG());
    return fir::factory::Complex{builder, loc}.createComplex(KIND, realPart,
                                                             imagPart);
  } else /*constexpr*/ {
    llvm_unreachable("unhandled constant");
  }
}

/// Create fir::string_lit from a scalar character constant.
template <int KIND>
static fir::StringLitOp
createStringLitOp(fir::FirOpBuilder &builder, mlir::Location loc,
                  const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Character, KIND>> &value,
                  [[maybe_unused]] int64_t len) {
  if constexpr (KIND == 1) {
    assert(value.size() == static_cast<std::uint64_t>(len));
    return builder.createStringLitOp(loc, value);
  } else {
    using ET = typename std::decay_t<decltype(value)>::value_type;
    fir::CharacterType type =
        fir::CharacterType::get(builder.getContext(), KIND, len);
    mlir::MLIRContext *context = builder.getContext();
    std::int64_t size = static_cast<std::int64_t>(value.size());
    mlir::ShapedType shape = mlir::RankedTensorType::get(
        llvm::ArrayRef<std::int64_t>{size},
        mlir::IntegerType::get(builder.getContext(), sizeof(ET) * 8));
    auto denseAttr = mlir::DenseElementsAttr::get(
        shape, llvm::ArrayRef<ET>{value.data(), value.size()});
    auto denseTag = mlir::StringAttr::get(context, fir::StringLitOp::xlist());
    mlir::NamedAttribute dataAttr(denseTag, denseAttr);
    auto sizeTag = mlir::StringAttr::get(context, fir::StringLitOp::size());
    mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(len));
    llvm::SmallVector<mlir::NamedAttribute> attrs = {dataAttr, sizeAttr};
    return builder.create<fir::StringLitOp>(
        loc, llvm::ArrayRef<mlir::Type>{type}, std::nullopt, attrs);
  }
}

/// Convert a scalar literal CHARACTER to IR.
template <int KIND>
static mlir::Value
genScalarLit(fir::FirOpBuilder &builder, mlir::Location loc,
             const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                 Fortran::common::TypeCategory::Character, KIND>> &value,
             int64_t len, bool outlineInReadOnlyMemory) {
  // When in an initializer context, construct the literal op itself and do
  // not construct another constant object in rodata.
  if (!outlineInReadOnlyMemory)
    return createStringLitOp<KIND>(builder, loc, value, len);

  // Otherwise, the string is in a plain old expression so "outline" the value
  // in read only data by hash consing it to a constant literal object.

  // ASCII global constants are created using an mlir string attribute.
  if constexpr (KIND == 1) {
    return fir::getBase(fir::factory::createStringLiteral(builder, loc, value));
  }

  auto size = builder.getKindMap().getCharacterBitsize(KIND) / 8 * value.size();
  llvm::StringRef strVal(reinterpret_cast<const char *>(value.c_str()), size);
  std::string globalName = fir::factory::uniqueCGIdent("cl", strVal);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  fir::CharacterType type =
      fir::CharacterType::get(builder.getContext(), KIND, len);
  if (!global)
    global = builder.createGlobalConstant(
        loc, type, globalName,
        [&](fir::FirOpBuilder &builder) {
          fir::StringLitOp str =
              createStringLitOp<KIND>(builder, loc, value, len);
          builder.create<fir::HasValueOp>(loc, str);
        },
        builder.createLinkOnceLinkage());
  return builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                       global.getSymbol());
}

/// Create an evaluate::Constant<T> array to a fir.array<> value
/// built with a chain of fir.insert or fir.insert_on_range operations.
/// This is intended to be called when building the body of a fir.global.
template <Fortran::common::TypeCategory TC, int KIND>
static mlir::Value genInlinedArrayLit(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type arrayTy,
    const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>> &con) {
  mlir::IndexType idxTy = builder.getIndexType();
  Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
  auto createIdx = [&]() {
    llvm::SmallVector<mlir::Attribute> idx;
    for (size_t i = 0; i < subscripts.size(); ++i)
      idx.push_back(
          builder.getIntegerAttr(idxTy, subscripts[i] - con.lbounds()[i]));
    return idx;
  };
  mlir::Value array = builder.create<fir::UndefOp>(loc, arrayTy);
  if (Fortran::evaluate::GetSize(con.shape()) == 0)
    return array;
  if constexpr (TC == Fortran::common::TypeCategory::Character) {
    do {
      mlir::Value elementVal =
          genScalarLit<KIND>(builder, loc, con.At(subscripts), con.LEN(),
                             /*outlineInReadOnlyMemory=*/false);
      array = builder.create<fir::InsertValueOp>(
          loc, arrayTy, array, elementVal, builder.getArrayAttr(createIdx()));
    } while (con.IncrementSubscripts(subscripts));
  } else {
    llvm::SmallVector<mlir::Attribute> rangeStartIdx;
    uint64_t rangeSize = 0;
    mlir::Type eleTy = arrayTy.cast<fir::SequenceType>().getEleTy();
    do {
      auto getElementVal = [&]() {
        return builder.createConvert(
            loc, eleTy,
            genScalarLit<TC, KIND>(builder, loc, con.At(subscripts)));
      };
      Fortran::evaluate::ConstantSubscripts nextSubscripts = subscripts;
      bool nextIsSame = con.IncrementSubscripts(nextSubscripts) &&
                        con.At(subscripts) == con.At(nextSubscripts);
      if (!rangeSize && !nextIsSame) { // single (non-range) value
        array = builder.create<fir::InsertValueOp>(
            loc, arrayTy, array, getElementVal(),
            builder.getArrayAttr(createIdx()));
      } else if (!rangeSize) { // start a range
        rangeStartIdx = createIdx();
        rangeSize = 1;
      } else if (nextIsSame) { // expand a range
        ++rangeSize;
      } else { // end a range
        llvm::SmallVector<int64_t> rangeBounds;
        llvm::SmallVector<mlir::Attribute> idx = createIdx();
        for (size_t i = 0; i < idx.size(); ++i) {
          rangeBounds.push_back(rangeStartIdx[i]
                                    .cast<mlir::IntegerAttr>()
                                    .getValue()
                                    .getSExtValue());
          rangeBounds.push_back(
              idx[i].cast<mlir::IntegerAttr>().getValue().getSExtValue());
        }
        array = builder.create<fir::InsertOnRangeOp>(
            loc, arrayTy, array, getElementVal(),
            builder.getIndexVectorAttr(rangeBounds));
        rangeSize = 0;
      }
    } while (con.IncrementSubscripts(subscripts));
  }
  return array;
}

/// Convert an evaluate::Constant<T> array into a fir.ref<fir.array<>> value
/// that points to the storage of a fir.global in read only memory and is
/// initialized with the value of the constant.
/// This should not be called while generating the body of a fir.global.
template <Fortran::common::TypeCategory TC, int KIND>
static mlir::Value genOutlineArrayLit(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type arrayTy,
    const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
        &constant) {
  std::string globalName = Fortran::lower::mangle::mangleArrayLiteral(constant);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  if (!global) {
    // Using a dense attribute for the initial value instead of creating an
    // intialization body speeds up MLIR/LLVM compilation, but this is not
    // always possible.
    if constexpr (TC == Fortran::common::TypeCategory::Logical ||
                  TC == Fortran::common::TypeCategory::Integer ||
                  TC == Fortran::common::TypeCategory::Real) {
      global = DenseGlobalBuilder::tryCreating(
          builder, loc, arrayTy, globalName, builder.createInternalLinkage(),
          true, constant);
    }
    if (!global)
      global = builder.createGlobalConstant(
          loc, arrayTy, globalName,
          [&](fir::FirOpBuilder &builder) {
            mlir::Value result =
                genInlinedArrayLit(builder, loc, arrayTy, constant);
            builder.create<fir::HasValueOp>(loc, result);
          },
          builder.createInternalLinkage());
  }
  return builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                       global.getSymbol());
}

/// Convert an evaluate::Constant<T> array into an fir::ExtendedValue.
template <Fortran::common::TypeCategory TC, int KIND>
static fir::ExtendedValue genArrayLit(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>> &con,
    bool outlineInReadOnlyMemory) {
  Fortran::evaluate::ConstantSubscript size =
      Fortran::evaluate::GetSize(con.shape());
  if (size > std::numeric_limits<std::uint32_t>::max())
    // llvm::SmallVector has limited size
    TODO(loc, "Creation of very large array constants");
  fir::SequenceType::Shape shape(con.shape().begin(), con.shape().end());
  llvm::SmallVector<std::int64_t> typeParams;
  if constexpr (TC == Fortran::common::TypeCategory::Character)
    typeParams.push_back(con.LEN());
  mlir::Type eleTy =
      Fortran::lower::getFIRType(builder.getContext(), TC, KIND, typeParams);
  auto arrayTy = fir::SequenceType::get(shape, eleTy);
  mlir::Value array = outlineInReadOnlyMemory
                          ? genOutlineArrayLit(builder, loc, arrayTy, con)
                          : genInlinedArrayLit(builder, loc, arrayTy, con);

  mlir::IndexType idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> extents;
  for (auto extent : shape)
    extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
  // Convert  lower bounds if they are not all ones.
  llvm::SmallVector<mlir::Value> lbounds;
  if (llvm::any_of(con.lbounds(), [](auto lb) { return lb != 1; }))
    for (auto lb : con.lbounds())
      lbounds.push_back(builder.createIntegerConstant(loc, idxTy, lb));

  if constexpr (TC == Fortran::common::TypeCategory::Character) {
    mlir::Value len = builder.createIntegerConstant(loc, idxTy, con.LEN());
    return fir::CharArrayBoxValue{array, len, extents, lbounds};
  } else {
    return fir::ArrayBoxValue{array, extents, lbounds};
  }
}

template <Fortran::common::TypeCategory TC, int KIND>
fir::ExtendedValue
Fortran::lower::ConstantBuilder<Fortran::evaluate::Type<TC, KIND>>::gen(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
        &constant,
    bool outlineBigConstantsInReadOnlyMemory) {
  if (constant.Rank() > 0)
    return genArrayLit<TC, KIND>(builder, loc, constant,
                                 outlineBigConstantsInReadOnlyMemory);
  std::optional<Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>>
      opt = constant.GetScalarValue();
  assert(opt.has_value() && "constant has no value");
  if constexpr (TC == Fortran::common::TypeCategory::Character) {
    auto value = genScalarLit<KIND>(builder, loc, opt.value(), constant.LEN(),
                                    outlineBigConstantsInReadOnlyMemory);
    mlir::Value len = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), constant.LEN());
    return fir::CharBoxValue{value, len};
  } else {
    return genScalarLit<TC, KIND>(builder, loc, opt.value());
  }
}

using namespace Fortran::evaluate;
FOR_EACH_INTRINSIC_KIND(template class Fortran::lower::ConstantBuilder, )
