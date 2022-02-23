//===-- ConvertType.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertType.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

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

static mlir::Type genComplexType(mlir::MLIRContext *context, int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Complex, KIND))
    return fir::ComplexType::get(context, KIND);
  return {};
}

static mlir::Type genFIRType(mlir::MLIRContext *context,
                             Fortran::common::TypeCategory tc, int kind) {
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
    TODO_NOLOC("genFIRType Character");
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
class TypeBuilder {
public:
  TypeBuilder(Fortran::lower::AbstractConverter &converter)
      : converter{converter}, context{&converter.getMLIRContext()} {}

  mlir::Type genExprType(const Fortran::lower::SomeExpr &expr) {
    std::optional<Fortran::evaluate::DynamicType> dynamicType = expr.GetType();
    if (!dynamicType)
      return genTypelessExprType(expr);
    Fortran::common::TypeCategory category = dynamicType->category();

    mlir::Type baseType;
    if (category == Fortran::common::TypeCategory::Derived) {
      TODO(converter.getCurrentLocation(), "genExprType derived");
    } else {
      // LOGICAL, INTEGER, REAL, COMPLEX, CHARACTER
      baseType = genFIRType(context, category, dynamicType->kind());
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
              TODO(converter.getCurrentLocation(),
                   "genTypelessExprType ProcedureDesignator");
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
    if (const Fortran::semantics::DeclTypeSpec *type = ultimate.GetType()) {
      if (const Fortran::semantics::IntrinsicTypeSpec *tySpec =
              type->AsIntrinsic()) {
        int kind = toInt64(Fortran::common::Clone(tySpec->kind())).value();
        ty = genFIRType(context, tySpec->category(), kind);
      } else if (type->IsPolymorphic()) {
        TODO(loc, "genSymbolType polymorphic types");
      } else if (type->AsDerived()) {
        TODO(loc, "genSymbolType derived type");
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

  mlir::Type genVariableType(const Fortran::lower::pft::Variable &var) {
    return genSymbolType(var.getSymbol(), var.isHeapAlloc(), var.isPointer());
  }

private:
  Fortran::lower::AbstractConverter &converter;
  mlir::MLIRContext *context;
};

} // namespace

mlir::Type Fortran::lower::getFIRType(mlir::MLIRContext *context,
                                      Fortran::common::TypeCategory tc,
                                      int kind) {
  return genFIRType(context, tc, kind);
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
