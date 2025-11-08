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
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/BuiltinModules.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Todo.h"

#include <algorithm>

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
    if constexpr (KIND <= 8)
      return builder.getIntegerAttr(type, value.ToInt64());
    else {
      static_assert(KIND <= 16, "integers with KIND > 16 are not supported");
      return builder.getIntegerAttr(
          type, llvm::APInt(KIND * 8,
                            {value.ToUInt64(), value.SHIFTR(64).ToUInt64()}));
    }
  } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
    return builder.getIntegerAttr(type, value.IsTrue());
  } else {
    auto getFloatAttr = [&](const auto &value, mlir::Type type) {
      std::string str = value.DumpHexadecimal();
      auto floatVal =
          consAPFloat(builder.getKindMap().getFloatSemantics(KIND), str);
      return builder.getFloatAttr(type, floatVal);
    };

    if constexpr (TC == Fortran::common::TypeCategory::Real) {
      return getFloatAttr(value, type);
    } else {
      static_assert(TC == Fortran::common::TypeCategory::Complex,
                    "type values cannot be converted to attributes");
      mlir::Type eleTy = mlir::cast<mlir::ComplexType>(type).getElementType();
      llvm::SmallVector<mlir::Attribute, 2> attrs = {
          getFloatAttr(value.REAL(), eleTy),
          getFloatAttr(value.AIMAG(), eleTy)};
      return builder.getArrayAttr(attrs);
    }
  }
  return {};
}

namespace {
/// Helper class to lower an array constant to a global with an MLIR dense
/// attribute.
///
/// If we have an array of integer, real, complex, or logical, then we can
/// create a global array with the dense attribute.
///
/// The mlir tensor type can only handle integer, real, complex, or logical.
/// It does not currently support nested structures.
class DenseGlobalBuilder {
public:
  static fir::GlobalOp tryCreating(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Type symTy,
                                   llvm::StringRef globalName,
                                   mlir::StringAttr linkage, bool isConst,
                                   const Fortran::lower::SomeExpr &initExpr,
                                   cuf::DataAttributeAttr dataAttr) {
    DenseGlobalBuilder globalBuilder;
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Expr<Fortran::evaluate::SomeLogical> &
                    x) { globalBuilder.tryConvertingToAttributes(builder, x); },
            [&](const Fortran::evaluate::Expr<Fortran::evaluate::SomeInteger> &
                    x) { globalBuilder.tryConvertingToAttributes(builder, x); },
            [&](const Fortran::evaluate::Expr<Fortran::evaluate::SomeReal> &x) {
              globalBuilder.tryConvertingToAttributes(builder, x);
            },
            [&](const Fortran::evaluate::Expr<Fortran::evaluate::SomeComplex> &
                    x) { globalBuilder.tryConvertingToAttributes(builder, x); },
            [](const auto &) {},
        },
        initExpr.u);
    return globalBuilder.tryCreatingGlobal(builder, loc, symTy, globalName,
                                           linkage, isConst, dataAttr);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  static fir::GlobalOp tryCreating(
      fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type symTy,
      llvm::StringRef globalName, mlir::StringAttr linkage, bool isConst,
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &constant,
      cuf::DataAttributeAttr dataAttr) {
    DenseGlobalBuilder globalBuilder;
    globalBuilder.tryConvertingToAttributes(builder, constant);
    return globalBuilder.tryCreatingGlobal(builder, loc, symTy, globalName,
                                           linkage, isConst, dataAttr);
  }

private:
  DenseGlobalBuilder() = default;

  /// Try converting an evaluate::Constant to a list of MLIR attributes.
  template <Fortran::common::TypeCategory TC, int KIND>
  void tryConvertingToAttributes(
      fir::FirOpBuilder &builder,
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &constant) {
    using Element =
        Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>;

    static_assert(TC != Fortran::common::TypeCategory::Character,
                  "must be numerical or logical");
    auto attrTc = TC == Fortran::common::TypeCategory::Logical
                      ? Fortran::common::TypeCategory::Integer
                      : TC;
    attributeElementType =
        Fortran::lower::getFIRType(builder.getContext(), attrTc, KIND, {});

    const std::vector<Element> &values = constant.values();
    auto sameElements = [&]() -> bool {
      if (values.empty())
        return false;

      return std::all_of(values.begin(), values.end(),
                         [&](const auto &v) { return v == values.front(); });
    };

    if (sameElements()) {
      auto attr = convertToAttribute<TC, KIND>(builder, values.front(),
                                               attributeElementType);
      attributes.assign(values.size(), attr);
      return;
    }

    for (auto element : values)
      attributes.push_back(
          convertToAttribute<TC, KIND>(builder, element, attributeElementType));
  }

  /// Try converting an evaluate::Expr to a list of MLIR attributes.
  template <typename SomeCat>
  void tryConvertingToAttributes(fir::FirOpBuilder &builder,
                                 const Fortran::evaluate::Expr<SomeCat> &expr) {
    Fortran::common::visit(
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
                                  mlir::StringAttr linkage, bool isConst,
                                  cuf::DataAttributeAttr dataAttr) const {
    // Not a "trivial" intrinsic constant array, or empty array.
    if (!attributeElementType || attributes.empty())
      return {};

    assert(mlir::isa<fir::SequenceType>(symTy) && "expecting an array global");
    auto arrTy = mlir::cast<fir::SequenceType>(symTy);
    llvm::SmallVector<int64_t> tensorShape(arrTy.getShape());
    std::reverse(tensorShape.begin(), tensorShape.end());
    auto tensorTy =
        mlir::RankedTensorType::get(tensorShape, attributeElementType);
    auto init = mlir::DenseElementsAttr::get(tensorTy, attributes);
    return builder.createGlobal(loc, symTy, globalName, linkage, init, isConst,
                                /*isTarget=*/false, dataAttr);
  }

  llvm::SmallVector<mlir::Attribute> attributes;
  mlir::Type attributeElementType;
};
} // namespace

fir::GlobalOp Fortran::lower::tryCreatingDenseGlobal(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type symTy,
    llvm::StringRef globalName, mlir::StringAttr linkage, bool isConst,
    const Fortran::lower::SomeExpr &initExpr, cuf::DataAttributeAttr dataAttr) {
  return DenseGlobalBuilder::tryCreating(builder, loc, symTy, globalName,
                                         linkage, isConst, initExpr, dataAttr);
}

//===----------------------------------------------------------------------===//
// Fortran::lower::convertConstant
// Lower a constant to a fir::ExtendedValue.
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
  if constexpr (TC == Fortran::common::TypeCategory::Integer ||
                TC == Fortran::common::TypeCategory::Unsigned) {
    // MLIR requires constants to be signless
    mlir::Type ty = Fortran::lower::getFIRType(
        builder.getContext(), Fortran::common::TypeCategory::Integer, KIND, {});
    if (KIND == 16) {
      auto bigInt = llvm::APInt(ty.getIntOrFloatBitWidth(),
                                TC == Fortran::common::TypeCategory::Unsigned
                                    ? value.UnsignedDecimal()
                                    : value.SignedDecimal(),
                                10);
      return mlir::arith::ConstantOp::create(
          builder, loc, ty, mlir::IntegerAttr::get(ty, bigInt));
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
    mlir::Value real = genScalarLit<Fortran::common::TypeCategory::Real, KIND>(
        builder, loc, value.REAL());
    mlir::Value imag = genScalarLit<Fortran::common::TypeCategory::Real, KIND>(
        builder, loc, value.AIMAG());
    return fir::factory::Complex{builder, loc}.createComplex(real, imag);
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
    return fir::StringLitOp::create(builder, loc,
                                    llvm::ArrayRef<mlir::Type>{type},
                                    mlir::ValueRange{}, attrs);
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
  std::string globalName = fir::factory::uniqueCGIdent(
      KIND == 1 ? "cl"s : "cl"s + std::to_string(KIND), strVal);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  fir::CharacterType type =
      fir::CharacterType::get(builder.getContext(), KIND, len);
  if (!global)
    global = builder.createGlobalConstant(
        loc, type, globalName,
        [&](fir::FirOpBuilder &builder) {
          fir::StringLitOp str =
              createStringLitOp<KIND>(builder, loc, value, len);
          fir::HasValueOp::create(builder, loc, str);
        },
        builder.createLinkOnceLinkage());
  return fir::AddrOfOp::create(builder, loc, global.resultType(),
                               global.getSymbol());
}

// Helper to generate StructureConstructor component values.
static fir::ExtendedValue
genConstantValue(Fortran::lower::AbstractConverter &converter,
                 mlir::Location loc,
                 const Fortran::lower::SomeExpr &constantExpr);

static mlir::Value genStructureComponentInit(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, const Fortran::lower::SomeExpr &expr,
    mlir::Value res) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  fir::RecordType recTy = mlir::cast<fir::RecordType>(res.getType());
  std::string name = converter.getRecordTypeFieldName(sym);
  mlir::Type componentTy = recTy.getType(name);
  auto fieldTy = fir::FieldType::get(recTy.getContext());
  assert(componentTy && "failed to retrieve component");
  // FIXME: type parameters must come from the derived-type-spec
  auto field =
      fir::FieldIndexOp::create(builder, loc, fieldTy, name, recTy,
                                /*typeParams=*/mlir::ValueRange{} /*TODO*/);

  if (Fortran::semantics::IsAllocatable(sym)) {
    if (!Fortran::evaluate::IsNullPointerOrAllocatable(&expr)) {
      fir::emitFatalError(loc, "constant structure constructor with an "
                               "allocatable component value that is not NULL");
    } else {
      // Handle NULL() initialization
      mlir::Value componentValue{
          fir::factory::createUnallocatedBox(builder, loc, componentTy, {})};
      componentValue = builder.createConvert(loc, componentTy, componentValue);

      return fir::InsertValueOp::create(
          builder, loc, recTy, res, componentValue,
          builder.getArrayAttr(field.getAttributes()));
    }
  }

  if (Fortran::semantics::IsPointer(sym)) {
    mlir::Value initialTarget;
    if (Fortran::semantics::IsProcedure(sym)) {
      if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(expr))
        initialTarget =
            fir::factory::createNullBoxProc(builder, loc, componentTy);
      else {
        Fortran::lower::SymMap globalOpSymMap;
        Fortran::lower::StatementContext stmtCtx;
        auto box{getBase(Fortran::lower::convertExprToAddress(
            loc, converter, expr, globalOpSymMap, stmtCtx))};
        initialTarget = builder.createConvert(loc, componentTy, box);
      }
    } else
      initialTarget = Fortran::lower::genInitialDataTarget(converter, loc,
                                                           componentTy, expr);
    res =
        fir::InsertValueOp::create(builder, loc, recTy, res, initialTarget,
                                   builder.getArrayAttr(field.getAttributes()));
    return res;
  }

  if (Fortran::lower::isDerivedTypeWithLenParameters(sym))
    TODO(loc, "component with length parameters in structure constructor");

  // Special handling for scalar c_ptr/c_funptr constants. The array constant
  // must fall through to genConstantValue() below.
  if (Fortran::semantics::IsBuiltinCPtr(sym) && sym.Rank() == 0 &&
      (Fortran::evaluate::GetLastSymbol(expr) ||
       Fortran::evaluate::IsNullPointer(&expr))) {
    // Builtin c_ptr and c_funptr have special handling because designators
    // and NULL() are handled as initial values for them as an extension
    // (otherwise only c_ptr_null/c_funptr_null are allowed and these are
    // replaced by structure constructors by semantics, so GetLastSymbol
    // returns nothing).

    // The Ev::Expr is an initializer that is a pointer target (e.g., 'x' or
    // NULL()) that must be inserted into an intermediate cptr record value's
    // address field, which ought to be an intptr_t on the target.
    mlir::Value addr = fir::getBase(
        Fortran::lower::genExtAddrInInitializer(converter, loc, expr));
    if (mlir::isa<fir::BoxProcType>(addr.getType()))
      addr = fir::BoxAddrOp::create(builder, loc, addr);
    assert((fir::isa_ref_type(addr.getType()) ||
            mlir::isa<mlir::FunctionType>(addr.getType())) &&
           "expect reference type for address field");
    assert(fir::isa_derived(componentTy) &&
           "expect C_PTR, C_FUNPTR to be a record");
    auto cPtrRecTy = mlir::cast<fir::RecordType>(componentTy);
    llvm::StringRef addrFieldName = Fortran::lower::builtin::cptrFieldName;
    mlir::Type addrFieldTy = cPtrRecTy.getType(addrFieldName);
    auto addrField = fir::FieldIndexOp::create(
        builder, loc, fieldTy, addrFieldName, componentTy,
        /*typeParams=*/mlir::ValueRange{});
    mlir::Value castAddr = builder.createConvert(loc, addrFieldTy, addr);
    auto undef = fir::UndefOp::create(builder, loc, componentTy);
    addr = fir::InsertValueOp::create(
        builder, loc, componentTy, undef, castAddr,
        builder.getArrayAttr(addrField.getAttributes()));
    res =
        fir::InsertValueOp::create(builder, loc, recTy, res, addr,
                                   builder.getArrayAttr(field.getAttributes()));
    return res;
  }

  mlir::Value val = fir::getBase(genConstantValue(converter, loc, expr));
  assert(!fir::isa_ref_type(val.getType()) && "expecting a constant value");
  mlir::Value castVal = builder.createConvert(loc, componentTy, val);
  res = fir::InsertValueOp::create(builder, loc, recTy, res, castVal,
                                   builder.getArrayAttr(field.getAttributes()));
  return res;
}

// Generate a StructureConstructor inlined (returns raw fir.type<T> value,
// not the address of a global constant).
static mlir::Value genInlinedStructureCtorLitImpl(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::evaluate::StructureConstructor &ctor, mlir::Type type) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto recTy = mlir::cast<fir::RecordType>(type);

  if (!converter.getLoweringOptions().getLowerToHighLevelFIR()) {
    mlir::Value res = fir::UndefOp::create(builder, loc, recTy);
    for (const auto &[sym, expr] : ctor.values()) {
      // Parent components need more work because they do not appear in the
      // fir.rec type.
      if (sym->test(Fortran::semantics::Symbol::Flag::ParentComp))
        TODO(loc, "parent component in structure constructor");
      res = genStructureComponentInit(converter, loc, sym, expr.value(), res);
    }
    return res;
  }

  auto fieldTy = fir::FieldType::get(recTy.getContext());
  mlir::Value res{};
  // When the first structure component values belong to some parent type PT
  // and the next values belong to a type extension ET, a new undef for ET must
  // be created and the previous PT value inserted into it. There may
  // be empty parent types in between ET and PT, hence the list and while loop.
  auto insertParentValueIntoExtension = [&](mlir::Type typeExtension) {
    assert(res && "res must be set");
    llvm::SmallVector<mlir::Type> parentTypes = {typeExtension};
    while (true) {
      fir::RecordType last = mlir::cast<fir::RecordType>(parentTypes.back());
      mlir::Type next =
          last.getType(0); // parent components are first in HLFIR.
      if (next != res.getType())
        parentTypes.push_back(next);
      else
        break;
    }
    for (mlir::Type parentType : llvm::reverse(parentTypes)) {
      auto undef = fir::UndefOp::create(builder, loc, parentType);
      fir::RecordType parentRecTy = mlir::cast<fir::RecordType>(parentType);
      auto field = fir::FieldIndexOp::create(
          builder, loc, fieldTy, parentRecTy.getTypeList()[0].first, parentType,
          /*typeParams=*/mlir::ValueRange{} /*TODO*/);
      res = fir::InsertValueOp::create(
          builder, loc, parentRecTy, undef, res,
          builder.getArrayAttr(field.getAttributes()));
    }
  };

  const Fortran::semantics::DerivedTypeSpec *curentType = nullptr;
  for (const auto &[sym, expr] : ctor.values()) {
    const Fortran::semantics::DerivedTypeSpec *componentParentType =
        sym->owner().derivedTypeSpec();
    assert(componentParentType && "failed to retrieve component parent type");
    if (!res) {
      mlir::Type parentType = converter.genType(*componentParentType);
      curentType = componentParentType;
      res = fir::UndefOp::create(builder, loc, parentType);
    } else if (*componentParentType != *curentType) {
      mlir::Type parentType = converter.genType(*componentParentType);
      insertParentValueIntoExtension(parentType);
      curentType = componentParentType;
    }
    res = genStructureComponentInit(converter, loc, sym, expr.value(), res);
  }

  if (!res) // structure constructor for empty type.
    return fir::UndefOp::create(builder, loc, recTy);

  // The last component may belong to a parent type.
  if (res.getType() != recTy)
    insertParentValueIntoExtension(recTy);
  return res;
}

static mlir::Value genScalarLit(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::evaluate::Scalar<Fortran::evaluate::SomeDerived> &value,
    mlir::Type eleTy, bool outlineBigConstantsInReadOnlyMemory) {
  if (!outlineBigConstantsInReadOnlyMemory)
    return genInlinedStructureCtorLitImpl(converter, loc, value, eleTy);
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto expr = std::make_unique<Fortran::lower::SomeExpr>(toEvExpr(
      Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived>(value)));
  llvm::StringRef globalName =
      converter.getUniqueLitName(loc, std::move(expr), eleTy);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  if (!global) {
    global = builder.createGlobalConstant(
        loc, eleTy, globalName,
        [&](fir::FirOpBuilder &builder) {
          mlir::Value result =
              genInlinedStructureCtorLitImpl(converter, loc, value, eleTy);
          fir::HasValueOp::create(builder, loc, result);
        },
        builder.createInternalLinkage());
  }
  return fir::AddrOfOp::create(builder, loc, global.resultType(),
                               global.getSymbol());
}

/// Create an evaluate::Constant<T> array to a fir.array<> value
/// built with a chain of fir.insert or fir.insert_on_range operations.
/// This is intended to be called when building the body of a fir.global.
template <typename T>
static mlir::Value
genInlinedArrayLit(Fortran::lower::AbstractConverter &converter,
                   mlir::Location loc, mlir::Type arrayTy,
                   const Fortran::evaluate::Constant<T> &con) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
  auto createIdx = [&]() {
    llvm::SmallVector<mlir::Attribute> idx;
    for (size_t i = 0; i < subscripts.size(); ++i)
      idx.push_back(
          builder.getIntegerAttr(idxTy, subscripts[i] - con.lbounds()[i]));
    return idx;
  };
  mlir::Value array = fir::UndefOp::create(builder, loc, arrayTy);
  if (Fortran::evaluate::GetSize(con.shape()) == 0)
    return array;
  if constexpr (T::category == Fortran::common::TypeCategory::Character) {
    do {
      mlir::Value elementVal =
          genScalarLit<T::kind>(builder, loc, con.At(subscripts), con.LEN(),
                                /*outlineInReadOnlyMemory=*/false);
      array =
          fir::InsertValueOp::create(builder, loc, arrayTy, array, elementVal,
                                     builder.getArrayAttr(createIdx()));
    } while (con.IncrementSubscripts(subscripts));
  } else if constexpr (T::category == Fortran::common::TypeCategory::Derived) {
    do {
      mlir::Type eleTy =
          mlir::cast<fir::SequenceType>(arrayTy).getElementType();
      mlir::Value elementVal =
          genScalarLit(converter, loc, con.At(subscripts), eleTy,
                       /*outlineInReadOnlyMemory=*/false);
      array =
          fir::InsertValueOp::create(builder, loc, arrayTy, array, elementVal,
                                     builder.getArrayAttr(createIdx()));
    } while (con.IncrementSubscripts(subscripts));
  } else {
    llvm::SmallVector<mlir::Attribute> rangeStartIdx;
    uint64_t rangeSize = 0;
    mlir::Type eleTy = mlir::cast<fir::SequenceType>(arrayTy).getElementType();
    do {
      auto getElementVal = [&]() {
        return builder.createConvert(loc, eleTy,
                                     genScalarLit<T::category, T::kind>(
                                         builder, loc, con.At(subscripts)));
      };
      Fortran::evaluate::ConstantSubscripts nextSubscripts = subscripts;
      bool nextIsSame = con.IncrementSubscripts(nextSubscripts) &&
                        con.At(subscripts) == con.At(nextSubscripts);
      if (!rangeSize && !nextIsSame) { // single (non-range) value
        array = fir::InsertValueOp::create(builder, loc, arrayTy, array,
                                           getElementVal(),
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
          rangeBounds.push_back(mlir::cast<mlir::IntegerAttr>(rangeStartIdx[i])
                                    .getValue()
                                    .getSExtValue());
          rangeBounds.push_back(
              mlir::cast<mlir::IntegerAttr>(idx[i]).getValue().getSExtValue());
        }
        array = fir::InsertOnRangeOp::create(
            builder, loc, arrayTy, array, getElementVal(),
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
template <typename T>
static mlir::Value
genOutlineArrayLit(Fortran::lower::AbstractConverter &converter,
                   mlir::Location loc, mlir::Type arrayTy,
                   const Fortran::evaluate::Constant<T> &constant) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type eleTy = mlir::cast<fir::SequenceType>(arrayTy).getElementType();
  llvm::StringRef globalName = converter.getUniqueLitName(
      loc, std::make_unique<Fortran::lower::SomeExpr>(toEvExpr(constant)),
      eleTy);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  if (!global) {
    // Using a dense attribute for the initial value instead of creating an
    // intialization body speeds up MLIR/LLVM compilation, but this is not
    // always possible.
    if constexpr (T::category == Fortran::common::TypeCategory::Logical ||
                  T::category == Fortran::common::TypeCategory::Integer ||
                  T::category == Fortran::common::TypeCategory::Real ||
                  T::category == Fortran::common::TypeCategory::Complex) {
      global = DenseGlobalBuilder::tryCreating(
          builder, loc, arrayTy, globalName, builder.createInternalLinkage(),
          true, constant, {});
    }
    if (!global)
      // If the number of elements of the array is huge, the compilation may
      // use a lot of memory and take a very long time to complete.
      // Empirical evidence shows that an array with 150000 elements of
      // complex type takes roughly 30 seconds to compile and uses 4GB of RAM,
      // on a modern machine.
      // It would be nice to add a driver switch to control the array size
      // after which flang should not continue to compile.
      global = builder.createGlobalConstant(
          loc, arrayTy, globalName,
          [&](fir::FirOpBuilder &builder) {
            mlir::Value result =
                genInlinedArrayLit(converter, loc, arrayTy, constant);
            fir::HasValueOp::create(builder, loc, result);
          },
          builder.createInternalLinkage());
  }
  return fir::AddrOfOp::create(builder, loc, global.resultType(),
                               global.getSymbol());
}

/// Convert an evaluate::Constant<T> array into an fir::ExtendedValue.
template <typename T>
static fir::ExtendedValue
genArrayLit(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
            const Fortran::evaluate::Constant<T> &con,
            bool outlineInReadOnlyMemory) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::evaluate::ConstantSubscript size =
      Fortran::evaluate::GetSize(con.shape());
  if (size > std::numeric_limits<std::uint32_t>::max())
    // llvm::SmallVector has limited size
    TODO(loc, "Creation of very large array constants");
  fir::SequenceType::Shape shape(con.shape().begin(), con.shape().end());
  llvm::SmallVector<std::int64_t> typeParams;
  if constexpr (T::category == Fortran::common::TypeCategory::Character)
    typeParams.push_back(con.LEN());
  mlir::Type eleTy;
  if constexpr (T::category == Fortran::common::TypeCategory::Derived)
    eleTy = Fortran::lower::translateDerivedTypeToFIRType(
        converter, con.GetType().GetDerivedTypeSpec());
  else
    eleTy = Fortran::lower::getFIRType(builder.getContext(), T::category,
                                       T::kind, typeParams);
  auto arrayTy = fir::SequenceType::get(shape, eleTy);
  mlir::Value array = outlineInReadOnlyMemory
                          ? genOutlineArrayLit(converter, loc, arrayTy, con)
                          : genInlinedArrayLit(converter, loc, arrayTy, con);

  mlir::IndexType idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> extents;
  for (auto extent : shape)
    extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
  // Convert  lower bounds if they are not all ones.
  llvm::SmallVector<mlir::Value> lbounds;
  if (llvm::any_of(con.lbounds(), [](auto lb) { return lb != 1; }))
    for (auto lb : con.lbounds())
      lbounds.push_back(builder.createIntegerConstant(loc, idxTy, lb));

  if constexpr (T::category == Fortran::common::TypeCategory::Character) {
    mlir::Value len = builder.createIntegerConstant(loc, idxTy, con.LEN());
    return fir::CharArrayBoxValue{array, len, extents, lbounds};
  } else {
    return fir::ArrayBoxValue{array, extents, lbounds};
  }
}

template <typename T>
fir::ExtendedValue Fortran::lower::ConstantBuilder<T>::gen(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::evaluate::Constant<T> &constant,
    bool outlineBigConstantsInReadOnlyMemory) {
  if (constant.Rank() > 0)
    return genArrayLit(converter, loc, constant,
                       outlineBigConstantsInReadOnlyMemory);
  std::optional<Fortran::evaluate::Scalar<T>> opt = constant.GetScalarValue();
  assert(opt.has_value() && "constant has no value");
  if constexpr (T::category == Fortran::common::TypeCategory::Character) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    auto value =
        genScalarLit<T::kind>(builder, loc, opt.value(), constant.LEN(),
                              outlineBigConstantsInReadOnlyMemory);
    mlir::Value len = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), constant.LEN());
    return fir::CharBoxValue{value, len};
  } else if constexpr (T::category == Fortran::common::TypeCategory::Derived) {
    mlir::Type eleTy = Fortran::lower::translateDerivedTypeToFIRType(
        converter, opt->GetType().GetDerivedTypeSpec());
    return genScalarLit(converter, loc, *opt, eleTy,
                        outlineBigConstantsInReadOnlyMemory);
  } else {
    return genScalarLit<T::category, T::kind>(converter.getFirOpBuilder(), loc,
                                              opt.value());
  }
}

static fir::ExtendedValue
genConstantValue(Fortran::lower::AbstractConverter &converter,
                 mlir::Location loc,
                 const Fortran::evaluate::Expr<Fortran::evaluate::SomeDerived>
                     &constantExpr) {
  if (const auto *constant = std::get_if<
          Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived>>(
          &constantExpr.u))
    return Fortran::lower::convertConstant(converter, loc, *constant,
                                           /*outline=*/false);
  if (const auto *structCtor =
          std::get_if<Fortran::evaluate::StructureConstructor>(&constantExpr.u))
    return Fortran::lower::genInlinedStructureCtorLit(converter, loc,
                                                      *structCtor);
  fir::emitFatalError(loc, "not a constant derived type expression");
}

template <Fortran::common::TypeCategory TC, int KIND>
static fir::ExtendedValue genConstantValue(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::evaluate::Expr<Fortran::evaluate::Type<TC, KIND>>
        &constantExpr) {
  using T = Fortran::evaluate::Type<TC, KIND>;
  if (const auto *constant =
          std::get_if<Fortran::evaluate::Constant<T>>(&constantExpr.u))
    return Fortran::lower::convertConstant(converter, loc, *constant,
                                           /*outline=*/false);
  fir::emitFatalError(loc, "not an evaluate::Constant<T>");
}

static fir::ExtendedValue
genConstantValue(Fortran::lower::AbstractConverter &converter,
                 mlir::Location loc,
                 const Fortran::lower::SomeExpr &constantExpr) {
  return Fortran::common::visit(
      [&](const auto &x) -> fir::ExtendedValue {
        using T = std::decay_t<decltype(x)>;
        if constexpr (Fortran::common::HasMember<
                          T, Fortran::lower::CategoryExpression>) {
          if constexpr (T::Result::category ==
                        Fortran::common::TypeCategory::Derived) {
            return genConstantValue(converter, loc, x);
          } else {
            return Fortran::common::visit(
                [&](const auto &preciseKind) {
                  return genConstantValue(converter, loc, preciseKind);
                },
                x.u);
          }
        } else {
          fir::emitFatalError(loc, "unexpected typeless constant value");
        }
      },
      constantExpr.u);
}

fir::ExtendedValue Fortran::lower::genInlinedStructureCtorLit(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::evaluate::StructureConstructor &ctor) {
  mlir::Type type = Fortran::lower::translateDerivedTypeToFIRType(
      converter, ctor.derivedTypeSpec());
  return genInlinedStructureCtorLitImpl(converter, loc, ctor, type);
}

using namespace Fortran::evaluate;
FOR_EACH_SPECIFIC_TYPE(template class Fortran::lower::ConstantBuilder, )
