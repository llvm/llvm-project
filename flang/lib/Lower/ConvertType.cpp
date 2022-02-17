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

template <typename A>
bool isConstant(const Fortran::evaluate::Expr<A> &e) {
  return Fortran::evaluate::IsConstantExpr(Fortran::lower::SomeExpr{e});
}

template <typename A>
int64_t toConstant(const Fortran::evaluate::Expr<A> &e) {
  auto opt = Fortran::evaluate::ToInt64(e);
  assert(opt.has_value() && "expression didn't resolve to a constant");
  return opt.value();
}

// one argument template, must be specialized
template <Fortran::common::TypeCategory TC>
mlir::Type genFIRType(mlir::MLIRContext *, int) {
  return {};
}

// two argument template
template <Fortran::common::TypeCategory TC, int KIND>
mlir::Type genFIRType(mlir::MLIRContext *context) {
  if constexpr (TC == Fortran::common::TypeCategory::Integer) {
    auto bits{Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer,
                                      KIND>::Scalar::bits};
    return mlir::IntegerType::get(context, bits);
  } else if constexpr (TC == Fortran::common::TypeCategory::Logical ||
                       TC == Fortran::common::TypeCategory::Character ||
                       TC == Fortran::common::TypeCategory::Complex) {
    return genFIRType<TC>(context, KIND);
  } else {
    return {};
  }
}

template <>
mlir::Type
genFIRType<Fortran::common::TypeCategory::Character>(mlir::MLIRContext *context,
                                                     int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Character, KIND))
    return fir::CharacterType::get(context, KIND, 1);
  return {};
}

namespace {

/// Discover the type of an Fortran::evaluate::Expr<T> and convert it to an
/// mlir::Type. The type returned may be an MLIR standard or FIR type.
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

  //===--------------------------------------------------------------------===//
  // Generate type entry points
  //===--------------------------------------------------------------------===//

  template <template <typename> typename A, Fortran::common::TypeCategory TC>
  mlir::Type gen(const A<Fortran::evaluate::SomeKind<TC>> &) {
    return genFIRType<TC>(context, defaultKind<TC>());
  }

  template <template <typename> typename A, Fortran::common::TypeCategory TC,
            int KIND>
  mlir::Type gen(const A<Fortran::evaluate::Type<TC, KIND>> &) {
    return genFIRType<TC, KIND>(context);
  }

  // breaks the conflict between A<Type<TC,KIND>> and Expr<B> deduction
  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Type
  gen(const Fortran::evaluate::Expr<Fortran::evaluate::Type<TC, KIND>> &) {
    return genFIRType<TC, KIND>(context);
  }

  // breaks the conflict between A<SomeKind<TC>> and Expr<B> deduction
  template <Fortran::common::TypeCategory TC>
  mlir::Type
  gen(const Fortran::evaluate::Expr<Fortran::evaluate::SomeKind<TC>> &expr) {
    return {};
  }

  template <typename A>
  mlir::Type gen(const Fortran::evaluate::Expr<A> &expr) {
    return {};
  }

  mlir::Type gen(const Fortran::evaluate::DataRef &dref) { return {}; }

  mlir::Type genVariableType(const Fortran::lower::pft::Variable &var) {
    return genSymbolType(var.getSymbol(), var.isHeapAlloc(), var.isPointer());
  }

  // non-template, category is runtime values, kind is defaulted
  mlir::Type genFIRTy(Fortran::common::TypeCategory tc) {
    return genFIRTy(tc, defaultKind(tc));
  }

  // non-template, arguments are runtime values
  mlir::Type genFIRTy(Fortran::common::TypeCategory tc, int kind) {
    switch (tc) {
    case Fortran::common::TypeCategory::Real:
      return genFIRType<Fortran::common::TypeCategory::Real>(context, kind);
    case Fortran::common::TypeCategory::Integer:
      return genFIRType<Fortran::common::TypeCategory::Integer>(context, kind);
    case Fortran::common::TypeCategory::Complex:
      return genFIRType<Fortran::common::TypeCategory::Complex>(context, kind);
    case Fortran::common::TypeCategory::Logical:
      return genFIRType<Fortran::common::TypeCategory::Logical>(context, kind);
    case Fortran::common::TypeCategory::Character:
      return genFIRType<Fortran::common::TypeCategory::Character>(context,
                                                                  kind);
    default:
      break;
    }
    llvm_unreachable("unhandled type category");
  }

private:
  //===--------------------------------------------------------------------===//
  // Generate type helpers
  //===--------------------------------------------------------------------===//

  mlir::Type gen(const Fortran::evaluate::ImpliedDoIndex &) {
    return genFIRType<Fortran::evaluate::ImpliedDoIndex::Result::category>(
        context, Fortran::evaluate::ImpliedDoIndex::Result::kind);
  }

  mlir::Type gen(const Fortran::evaluate::TypeParamInquiry &) {
    return genFIRType<Fortran::evaluate::TypeParamInquiry::Result::category>(
        context, Fortran::evaluate::TypeParamInquiry::Result::kind);
  }

  template <typename A>
  mlir::Type gen(const Fortran::evaluate::Relational<A> &) {
    return genFIRType<Fortran::common::TypeCategory::Logical, 1>(context);
  }

  // some sequence of `n` bytes
  mlir::Type gen(const Fortran::evaluate::StaticDataObject::Pointer &ptr) {
    mlir::Type byteTy{mlir::IntegerType::get(context, 8)};
    return fir::SequenceType::get(trivialShape(ptr->itemBytes()), byteTy);
  }

  mlir::Type gen(const Fortran::evaluate::Substring &ss) { return {}; }

  mlir::Type gen(const Fortran::evaluate::NullPointer &) {
    return genTypelessPtr();
  }
  mlir::Type gen(const Fortran::evaluate::ProcedureRef &) {
    return genTypelessPtr();
  }
  mlir::Type gen(const Fortran::evaluate::ProcedureDesignator &) {
    return genTypelessPtr();
  }
  mlir::Type gen(const Fortran::evaluate::BOZLiteralConstant &) {
    return genTypelessPtr();
  }
  mlir::Type gen(const Fortran::evaluate::ArrayRef &) {
    TODO_NOLOC("array ref");
  }
  mlir::Type gen(const Fortran::evaluate::CoarrayRef &) {
    TODO_NOLOC("coarray ref");
  }
  mlir::Type gen(const Fortran::evaluate::Component &) {
    TODO_NOLOC("component");
  }
  mlir::Type gen(const Fortran::evaluate::ComplexPart &) {
    TODO_NOLOC("complex part");
  }
  mlir::Type gen(const Fortran::evaluate::DescriptorInquiry &) {
    TODO_NOLOC("descriptor inquiry");
  }
  mlir::Type gen(const Fortran::evaluate::StructureConstructor &) {
    TODO_NOLOC("structure constructor");
  }

  fir::SequenceType::Shape genSeqShape(Fortran::semantics::SymbolRef symbol) {
    assert(symbol->IsObjectArray() && "unexpected symbol type");
    fir::SequenceType::Shape bounds;
    return seqShapeHelper(symbol, bounds);
  }

  fir::SequenceType::Shape genSeqShape(Fortran::semantics::SymbolRef symbol,
                                       fir::SequenceType::Extent charLen) {
    assert(symbol->IsObjectArray() && "unexpected symbol type");
    fir::SequenceType::Shape bounds;
    bounds.push_back(charLen);
    return seqShapeHelper(symbol, bounds);
  }

  //===--------------------------------------------------------------------===//
  // Other helper functions
  //===--------------------------------------------------------------------===//

  fir::SequenceType::Shape trivialShape(int size) {
    fir::SequenceType::Shape bounds;
    bounds.emplace_back(size);
    return bounds;
  }

  mlir::Type mkVoid() { return mlir::TupleType::get(context); }
  mlir::Type genTypelessPtr() { return fir::ReferenceType::get(mkVoid()); }

  template <Fortran::common::TypeCategory TC>
  int defaultKind() {
    return defaultKind(TC);
  }
  int defaultKind(Fortran::common::TypeCategory TC) { return 0; }

  fir::SequenceType::Shape seqShapeHelper(Fortran::semantics::SymbolRef symbol,
                                          fir::SequenceType::Shape &bounds) {
    auto &details = symbol->get<Fortran::semantics::ObjectEntityDetails>();
    const auto size = details.shape().size();
    for (auto &ss : details.shape()) {
      auto lb = ss.lbound();
      auto ub = ss.ubound();
      if (lb.isStar() && ub.isStar() && size == 1)
        return {}; // assumed rank
      if (lb.isExplicit() && ub.isExplicit()) {
        auto &lbv = lb.GetExplicit();
        auto &ubv = ub.GetExplicit();
        if (lbv.has_value() && ubv.has_value() && isConstant(lbv.value()) &&
            isConstant(ubv.value())) {
          bounds.emplace_back(toConstant(ubv.value()) -
                              toConstant(lbv.value()) + 1);
        } else {
          bounds.emplace_back(fir::SequenceType::getUnknownExtent());
        }
      } else {
        bounds.emplace_back(fir::SequenceType::getUnknownExtent());
      }
    }
    return bounds;
  }

  //===--------------------------------------------------------------------===//
  // Emit errors and warnings.
  //===--------------------------------------------------------------------===//

  mlir::InFlightDiagnostic emitError(const llvm::Twine &message) {
    return mlir::emitError(mlir::UnknownLoc::get(context), message);
  }

  mlir::InFlightDiagnostic emitWarning(const llvm::Twine &message) {
    return mlir::emitWarning(mlir::UnknownLoc::get(context), message);
  }

  //===--------------------------------------------------------------------===//

  Fortran::lower::AbstractConverter &converter;
  mlir::MLIRContext *context;
};

} // namespace

mlir::Type Fortran::lower::getFIRType(mlir::MLIRContext *context,
                                      Fortran::common::TypeCategory tc,
                                      int kind) {
  return genFIRType(context, tc, kind);
}

mlir::Type
Fortran::lower::getFIRType(Fortran::lower::AbstractConverter &converter,
                           Fortran::common::TypeCategory tc) {
  return TypeBuilder{converter}.genFIRTy(tc);
}

mlir::Type Fortran::lower::translateDataRefToFIRType(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::DataRef &dataRef) {
  return TypeBuilder{converter}.gen(dataRef);
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
  return genFIRType<Fortran::common::TypeCategory::Real>(context, kind);
}

mlir::Type Fortran::lower::getSequenceRefType(mlir::Type refType) {
  auto type{refType.dyn_cast<fir::ReferenceType>()};
  assert(type && "expected a reference type");
  auto elementType{type.getEleTy()};
  fir::SequenceType::Shape shape{fir::SequenceType::getUnknownExtent()};
  return fir::ReferenceType::get(fir::SequenceType::get(shape, elementType));
}
