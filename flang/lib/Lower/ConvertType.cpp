//===-- ConvertType.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertType.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-type"

//===--------------------------------------------------------------------===//
// Intrinsic type translation helpers
//===--------------------------------------------------------------------===//

static mlir::Type genRealType(mlir::MLIRContext *context, int kind) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Real, kind)) {
    switch (kind) {
    case 2:
      return mlir::FloatType::getF16(context);
    case 3:
      return mlir::FloatType::getBF16(context);
    case 4:
      return mlir::FloatType::getF32(context);
    case 8:
      return mlir::FloatType::getF64(context);
    case 10:
      return mlir::FloatType::getF80(context);
    case 16:
      return mlir::FloatType::getF128(context);
    }
  }
  llvm_unreachable("REAL type translation not implemented");
}

template <int KIND>
int getIntegerBits() {
  return Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer,
                                 KIND>::Scalar::bits;
}
static mlir::Type genIntegerType(mlir::MLIRContext *context, int kind) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Integer, kind)) {
    switch (kind) {
    case 1:
      return mlir::IntegerType::get(context, getIntegerBits<1>());
    case 2:
      return mlir::IntegerType::get(context, getIntegerBits<2>());
    case 4:
      return mlir::IntegerType::get(context, getIntegerBits<4>());
    case 8:
      return mlir::IntegerType::get(context, getIntegerBits<8>());
    case 16:
      return mlir::IntegerType::get(context, getIntegerBits<16>());
    }
  }
  llvm_unreachable("INTEGER kind not translated");
}

static mlir::Type genLogicalType(mlir::MLIRContext *context, int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Logical, KIND))
    return fir::LogicalType::get(context, KIND);
  return {};
}

static mlir::Type genCharacterType(
    mlir::MLIRContext *context, int KIND,
    Fortran::lower::LenParameterTy len = fir::CharacterType::unknownLen()) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Character, KIND))
    return fir::CharacterType::get(context, KIND, len);
  return {};
}

static mlir::Type genComplexType(mlir::MLIRContext *context, int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Complex, KIND))
    return fir::ComplexType::get(context, KIND);
  return {};
}

static mlir::Type
genFIRType(mlir::MLIRContext *context, Fortran::common::TypeCategory tc,
           int kind,
           llvm::ArrayRef<Fortran::lower::LenParameterTy> lenParameters) {
  switch (tc) {
  case Fortran::common::TypeCategory::Real:
    return genRealType(context, kind);
  case Fortran::common::TypeCategory::Integer:
    return genIntegerType(context, kind);
  case Fortran::common::TypeCategory::Complex:
    return genComplexType(context, kind);
  case Fortran::common::TypeCategory::Logical:
    return genLogicalType(context, kind);
  case Fortran::common::TypeCategory::Character:
    if (!lenParameters.empty())
      return genCharacterType(context, kind, lenParameters[0]);
    return genCharacterType(context, kind);
  default:
    break;
  }
  llvm_unreachable("unhandled type category");
}

//===--------------------------------------------------------------------===//
// Symbol and expression type translation
//===--------------------------------------------------------------------===//

/// TypeBuilder translates expression and symbol type taking into account
/// their shape and length parameters. For symbols, attributes such as
/// ALLOCATABLE or POINTER are reflected in the fir type.
/// It uses evaluate::DynamicType and evaluate::Shape when possible to
/// avoid re-implementing type/shape analysis here.
/// Do not use the FirOpBuilder from the AbstractConverter to get fir/mlir types
/// since it is not guaranteed to exist yet when we lower types.
namespace {
struct TypeBuilder {

  TypeBuilder(Fortran::lower::AbstractConverter &converter)
      : converter{converter}, context{&converter.getMLIRContext()} {}

  mlir::Type genExprType(const Fortran::lower::SomeExpr &expr) {
    std::optional<Fortran::evaluate::DynamicType> dynamicType = expr.GetType();
    if (!dynamicType)
      return genTypelessExprType(expr);
    Fortran::common::TypeCategory category = dynamicType->category();

    mlir::Type baseType;
    if (category == Fortran::common::TypeCategory::Derived) {
      baseType = genDerivedType(dynamicType->GetDerivedTypeSpec());
    } else {
      // LOGICAL, INTEGER, REAL, COMPLEX, CHARACTER
      llvm::SmallVector<Fortran::lower::LenParameterTy> params;
      translateLenParameters(params, category, expr);
      baseType = genFIRType(context, category, dynamicType->kind(), params);
    }
    std::optional<Fortran::evaluate::Shape> shapeExpr =
        Fortran::evaluate::GetShape(converter.getFoldingContext(), expr);
    fir::SequenceType::Shape shape;
    if (shapeExpr) {
      translateShape(shape, std::move(*shapeExpr));
    } else {
      // Shape static analysis cannot return something useful for the shape.
      // Use unknown extents.
      int rank = expr.Rank();
      if (rank < 0)
        TODO(converter.getCurrentLocation(),
             "Assumed rank expression type lowering");
      for (int dim = 0; dim < rank; ++dim)
        shape.emplace_back(fir::SequenceType::getUnknownExtent());
    }
    if (!shape.empty())
      return fir::SequenceType::get(shape, baseType);
    return baseType;
  }

  template <typename A>
  void translateShape(A &shape, Fortran::evaluate::Shape &&shapeExpr) {
    for (Fortran::evaluate::MaybeExtentExpr extentExpr : shapeExpr) {
      fir::SequenceType::Extent extent = fir::SequenceType::getUnknownExtent();
      if (std::optional<std::int64_t> constantExtent =
              toInt64(std::move(extentExpr)))
        extent = *constantExtent;
      shape.push_back(extent);
    }
  }

  template <typename A>
  std::optional<std::int64_t> toInt64(A &&expr) {
    return Fortran::evaluate::ToInt64(Fortran::evaluate::Fold(
        converter.getFoldingContext(), std::move(expr)));
  }

  mlir::Type genTypelessExprType(const Fortran::lower::SomeExpr &expr) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::BOZLiteralConstant &) -> mlir::Type {
              return mlir::NoneType::get(context);
            },
            [&](const Fortran::evaluate::NullPointer &) -> mlir::Type {
              return fir::ReferenceType::get(mlir::NoneType::get(context));
            },
            [&](const Fortran::evaluate::ProcedureDesignator &proc)
                -> mlir::Type {
              return Fortran::lower::translateSignature(proc, converter);
            },
            [&](const Fortran::evaluate::ProcedureRef &) -> mlir::Type {
              return mlir::NoneType::get(context);
            },
            [](const auto &x) -> mlir::Type {
              using T = std::decay_t<decltype(x)>;
              static_assert(!Fortran::common::HasMember<
                                T, Fortran::evaluate::TypelessExpression>,
                            "missing typeless expr handling in type lowering");
              llvm::report_fatal_error("not a typeless expression");
            },
        },
        expr.u);
  }

  mlir::Type genSymbolType(const Fortran::semantics::Symbol &symbol,
                           bool isAlloc = false, bool isPtr = false) {
    mlir::Location loc = converter.genLocation(symbol.name());
    mlir::Type ty;
    // If the symbol is not the same as the ultimate one (i.e, it is host or use
    // associated), all the symbol properties are the ones of the ultimate
    // symbol but the volatile and asynchronous attributes that may differ. To
    // avoid issues with helper functions that would not follow association
    // links, the fir type is built based on the ultimate symbol. This relies
    // on the fact volatile and asynchronous are not reflected in fir types.
    const Fortran::semantics::Symbol &ultimate = symbol.GetUltimate();
    if (Fortran::semantics::IsProcedurePointer(ultimate))
      TODO(loc, "procedure pointers");
    if (const Fortran::semantics::DeclTypeSpec *type = ultimate.GetType()) {
      if (const Fortran::semantics::IntrinsicTypeSpec *tySpec =
              type->AsIntrinsic()) {
        int kind = toInt64(Fortran::common::Clone(tySpec->kind())).value();
        llvm::SmallVector<Fortran::lower::LenParameterTy> params;
        translateLenParameters(params, tySpec->category(), ultimate);
        ty = genFIRType(context, tySpec->category(), kind, params);
      } else if (type->IsPolymorphic()) {
        TODO(loc, "[genSymbolType] polymorphic types");
      } else if (const Fortran::semantics::DerivedTypeSpec *tySpec =
                     type->AsDerived()) {
        ty = genDerivedType(*tySpec);
      } else {
        fir::emitFatalError(loc, "symbol's type must have a type spec");
      }
    } else {
      fir::emitFatalError(loc, "symbol must have a type");
    }
    if (ultimate.IsObjectArray()) {
      auto shapeExpr = Fortran::evaluate::GetShapeHelper{
          converter.getFoldingContext()}(ultimate);
      if (!shapeExpr)
        TODO(loc, "assumed rank symbol type lowering");
      fir::SequenceType::Shape shape;
      translateShape(shape, std::move(*shapeExpr));
      ty = fir::SequenceType::get(shape, ty);
    }

    if (Fortran::semantics::IsPointer(symbol))
      return fir::BoxType::get(fir::PointerType::get(ty));
    if (Fortran::semantics::IsAllocatable(symbol))
      return fir::BoxType::get(fir::HeapType::get(ty));
    // isPtr and isAlloc are variable that were promoted to be on the
    // heap or to be pointers, but they do not have Fortran allocatable
    // or pointer semantics, so do not use box for them.
    if (isPtr)
      return fir::PointerType::get(ty);
    if (isAlloc)
      return fir::HeapType::get(ty);
    return ty;
  }

  /// Does \p component has non deferred lower bounds that are not compile time
  /// constant 1.
  static bool componentHasNonDefaultLowerBounds(
      const Fortran::semantics::Symbol &component) {
    if (const auto *objDetails =
            component.detailsIf<Fortran::semantics::ObjectEntityDetails>())
      for (const Fortran::semantics::ShapeSpec &bounds : objDetails->shape())
        if (auto lb = bounds.lbound().GetExplicit())
          if (auto constant = Fortran::evaluate::ToInt64(*lb))
            if (!constant || *constant != 1)
              return true;
    return false;
  }

  mlir::Type genDerivedType(const Fortran::semantics::DerivedTypeSpec &tySpec) {
    std::vector<std::pair<std::string, mlir::Type>> ps;
    std::vector<std::pair<std::string, mlir::Type>> cs;
    const Fortran::semantics::Symbol &typeSymbol = tySpec.typeSymbol();
    if (mlir::Type ty = getTypeIfDerivedAlreadyInConstruction(typeSymbol))
      return ty;
    auto rec = fir::RecordType::get(context,
                                    Fortran::lower::mangle::mangleName(tySpec));
    // Maintain the stack of types for recursive references.
    derivedTypeInConstruction.emplace_back(typeSymbol, rec);

    // Gather the record type fields.
    // (1) The data components.
    for (const auto &field :
         Fortran::semantics::OrderedComponentIterator(tySpec)) {
      // Lowering is assuming non deferred component lower bounds are always 1.
      // Catch any situations where this is not true for now.
      if (componentHasNonDefaultLowerBounds(field))
        TODO(converter.genLocation(field.name()),
             "lowering derived type components with non default lower bounds");
      if (IsProcedure(field))
        TODO(converter.genLocation(field.name()), "procedure components");
      mlir::Type ty = genSymbolType(field);
      // Do not add the parent component (component of the parents are
      // added and should be sufficient, the parent component would
      // duplicate the fields).
      if (field.test(Fortran::semantics::Symbol::Flag::ParentComp))
        continue;
      cs.emplace_back(field.name().ToString(), ty);
    }

    // (2) The LEN type parameters.
    for (const auto &param :
         Fortran::semantics::OrderParameterDeclarations(typeSymbol))
      if (param->get<Fortran::semantics::TypeParamDetails>().attr() ==
          Fortran::common::TypeParamAttr::Len)
        ps.emplace_back(param->name().ToString(), genSymbolType(*param));

    rec.finalize(ps, cs);
    popDerivedTypeInConstruction();

    mlir::Location loc = converter.genLocation(typeSymbol.name());
    if (!ps.empty()) {
      // This type is a PDT (parametric derived type). Create the functions to
      // use for allocation, dereferencing, and address arithmetic here.
      TODO(loc, "parametrized derived types lowering");
    }
    LLVM_DEBUG(llvm::dbgs() << "derived type: " << rec << '\n');

    // Generate the type descriptor object if any
    if (const Fortran::semantics::Scope *derivedScope =
            tySpec.scope() ? tySpec.scope() : tySpec.typeSymbol().scope())
      if (const Fortran::semantics::Symbol *typeInfoSym =
              derivedScope->runtimeDerivedTypeDescription())
        converter.registerRuntimeTypeInfo(loc, *typeInfoSym);
    return rec;
  }

  // To get the character length from a symbol, make an fold a designator for
  // the symbol to cover the case where the symbol is an assumed length named
  // constant and its length comes from its init expression length.
  template <int Kind>
  fir::SequenceType::Extent
  getCharacterLengthHelper(const Fortran::semantics::Symbol &symbol) {
    using TC =
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, Kind>;
    auto designator = Fortran::evaluate::Fold(
        converter.getFoldingContext(),
        Fortran::evaluate::Expr<TC>{Fortran::evaluate::Designator<TC>{symbol}});
    if (auto len = toInt64(std::move(designator.LEN())))
      return *len;
    return fir::SequenceType::getUnknownExtent();
  }

  template <typename T>
  void translateLenParameters(
      llvm::SmallVectorImpl<Fortran::lower::LenParameterTy> &params,
      Fortran::common::TypeCategory category, const T &exprOrSym) {
    if (category == Fortran::common::TypeCategory::Character)
      params.push_back(getCharacterLength(exprOrSym));
    else if (category == Fortran::common::TypeCategory::Derived)
      TODO(converter.getCurrentLocation(),
           "lowering derived type length parameters");
    return;
  }
  Fortran::lower::LenParameterTy
  getCharacterLength(const Fortran::semantics::Symbol &symbol) {
    const Fortran::semantics::DeclTypeSpec *type = symbol.GetType();
    if (!type ||
        type->category() != Fortran::semantics::DeclTypeSpec::Character ||
        !type->AsIntrinsic())
      llvm::report_fatal_error("not a character symbol");
    int kind =
        toInt64(Fortran::common::Clone(type->AsIntrinsic()->kind())).value();
    switch (kind) {
    case 1:
      return getCharacterLengthHelper<1>(symbol);
    case 2:
      return getCharacterLengthHelper<2>(symbol);
    case 4:
      return getCharacterLengthHelper<4>(symbol);
    }
    llvm_unreachable("unknown character kind");
  }
  Fortran::lower::LenParameterTy
  getCharacterLength(const Fortran::lower::SomeExpr &expr) {
    // Do not use dynamic type length here. We would miss constant
    // lengths opportunities because dynamic type only has the length
    // if it comes from a declaration.
    auto charExpr =
        std::get<Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(
            expr.u);
    if (auto constantLen = toInt64(charExpr.LEN()))
      return *constantLen;
    return fir::SequenceType::getUnknownExtent();
  }

  mlir::Type genVariableType(const Fortran::lower::pft::Variable &var) {
    return genSymbolType(var.getSymbol(), var.isHeapAlloc(), var.isPointer());
  }

  /// Derived type can be recursive. That is, pointer components of a derived
  /// type `t` have type `t`. This helper returns `t` if it is already being
  /// lowered to avoid infinite loops.
  mlir::Type getTypeIfDerivedAlreadyInConstruction(
      const Fortran::lower::SymbolRef derivedSym) const {
    for (const auto &[sym, type] : derivedTypeInConstruction)
      if (sym == derivedSym)
        return type;
    return {};
  }

  void popDerivedTypeInConstruction() {
    assert(!derivedTypeInConstruction.empty());
    derivedTypeInConstruction.pop_back();
  }

  /// Stack derived type being processed to avoid infinite loops in case of
  /// recursive derived types. The depth of derived types is expected to be
  /// shallow (<10), so a SmallVector is sufficient.
  llvm::SmallVector<std::pair<const Fortran::lower::SymbolRef, mlir::Type>>
      derivedTypeInConstruction;
  Fortran::lower::AbstractConverter &converter;
  mlir::MLIRContext *context;
};
} // namespace

mlir::Type Fortran::lower::getFIRType(mlir::MLIRContext *context,
                                      Fortran::common::TypeCategory tc,
                                      int kind,
                                      llvm::ArrayRef<LenParameterTy> params) {
  return genFIRType(context, tc, kind, params);
}

mlir::Type Fortran::lower::translateDerivedTypeToFIRType(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::DerivedTypeSpec &tySpec) {
  return TypeBuilder{converter}.genDerivedType(tySpec);
}

mlir::Type Fortran::lower::translateSomeExprToFIRType(
    Fortran::lower::AbstractConverter &converter, const SomeExpr &expr) {
  return TypeBuilder{converter}.genExprType(expr);
}

mlir::Type Fortran::lower::translateSymbolToFIRType(
    Fortran::lower::AbstractConverter &converter, const SymbolRef symbol) {
  return TypeBuilder{converter}.genSymbolType(symbol);
}

mlir::Type Fortran::lower::translateVariableToFIRType(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  return TypeBuilder{converter}.genVariableType(var);
}

mlir::Type Fortran::lower::convertReal(mlir::MLIRContext *context, int kind) {
  return genRealType(context, kind);
}
