//===-- ConvertExpr.cpp ---------------------------------------------------===//
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

#include "flang/Lower/ConvertExpr.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/real.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-expr"

//===----------------------------------------------------------------------===//
// The composition and structure of Fortran::evaluate::Expr is defined in
// the various header files in include/flang/Evaluate. You are referred
// there for more information on these data structures. Generally speaking,
// these data structures are a strongly typed family of abstract data types
// that, composed as trees, describe the syntax of Fortran expressions.
//
// This part of the bridge can traverse these tree structures and lower them
// to the correct FIR representation in SSA form.
//===----------------------------------------------------------------------===//

/// Generate a load of a value from an address. Beware that this will lose
/// any dynamic type information for polymorphic entities (note that unlimited
/// polymorphic cannot be loaded and must not be provided here).
static fir::ExtendedValue genLoad(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  const fir::ExtendedValue &addr) {
  return addr.match(
      [](const fir::CharBoxValue &box) -> fir::ExtendedValue { return box; },
      [&](const fir::UnboxedValue &v) -> fir::ExtendedValue {
        if (fir::unwrapRefType(fir::getBase(v).getType())
                .isa<fir::RecordType>())
          return v;
        return builder.create<fir::LoadOp>(loc, fir::getBase(v));
      },
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        TODO(loc, "genLoad for MutableBoxValue");
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        TODO(loc, "genLoad for BoxValue");
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(
            loc, "attempting to load whole array or procedure address");
      });
}

namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ScalarExprLowering {
public:
  using ExtValue = fir::ExtendedValue;

  explicit ScalarExprLowering(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              Fortran::lower::SymMap &symMap)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, symMap{symMap} {}

  mlir::Location getLoc() { return location; }

  /// Generate an integral constant of `value`
  template <int KIND>
  mlir::Value genIntegerConstant(mlir::MLIRContext *context,
                                 std::int64_t value) {
    mlir::Type type =
        converter.genType(Fortran::common::TypeCategory::Integer, KIND);
    return builder.createIntegerConstant(getLoc(), type, value);
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genBoolConstant(bool value) {
    return builder.createBool(getLoc(), value);
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  ExtValue gen(Fortran::semantics::SymbolRef sym) {
    if (Fortran::lower::SymbolBox val = symMap.lookupSymbol(sym))
      return val.match([&val](auto &) { return val.toExtendedValue(); });
    LLVM_DEBUG(llvm::dbgs()
               << "unknown symbol: " << sym << "\nmap: " << symMap << '\n');
    fir::emitFatalError(getLoc(), "symbol is not mapped to any IR value");
  }

  ExtValue genLoad(const ExtValue &exv) {
    return ::genLoad(builder, getLoc(), exv);
  }

  ExtValue genval(Fortran::semantics::SymbolRef sym) {
    ExtValue var = gen(sym);
    if (const fir::UnboxedValue *s = var.getUnboxed())
      if (fir::isReferenceLike(s->getType()))
        return genLoad(*s);
    return var;
  }

  ExtValue genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO(getLoc(), "genval BOZ");
  }

  /// Return indirection to function designated in ProcedureDesignator.
  /// The type of the function indirection is not guaranteed to match the one
  /// of the ProcedureDesignator due to Fortran implicit typing rules.
  ExtValue genval(const Fortran::evaluate::ProcedureDesignator &proc) {
    TODO(getLoc(), "genval ProcedureDesignator");
  }

  ExtValue genval(const Fortran::evaluate::NullPointer &) {
    TODO(getLoc(), "genval NullPointer");
  }

  ExtValue genval(const Fortran::evaluate::StructureConstructor &ctor) {
    TODO(getLoc(), "genval StructureConstructor");
  }

  /// Lowering of an <i>ac-do-variable</i>, which is not a Symbol.
  ExtValue genval(const Fortran::evaluate::ImpliedDoIndex &var) {
    TODO(getLoc(), "genval ImpliedDoIndex");
  }

  ExtValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    TODO(getLoc(), "genval DescriptorInquiry");
  }

  ExtValue genval(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "genval TypeParamInquiry");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    TODO(getLoc(), "genval ComplexComponent");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    TODO(getLoc(), "genval Negate integer");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    TODO(getLoc(), "genval Negate real");
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    TODO(getLoc(), "genval Negate complex");
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  ExtValue genval(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type< \
                      Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) { \
    TODO(getLoc(), "genval GenBinEvOp");                                       \
  }

  GENBIN(Add, Integer, mlir::arith::AddIOp)
  GENBIN(Add, Real, mlir::arith::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::arith::SubIOp)
  GENBIN(Subtract, Real, mlir::arith::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::arith::MulIOp)
  GENBIN(Multiply, Real, mlir::arith::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::arith::DivSIOp)
  GENBIN(Divide, Real, mlir::arith::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    TODO(getLoc(), "genval Power");
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    TODO(getLoc(), "genval RealToInt");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    TODO(getLoc(), "genval ComplexConstructor");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    TODO(getLoc(), "genval Concat<KIND>");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    TODO(getLoc(), "genval Extremum<TC, KIND>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::SetLength<KIND> &x) {
    TODO(getLoc(), "genval SetLength<KIND>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    TODO(getLoc(), "genval integer comparison");
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    TODO(getLoc(), "genval real comparison");
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    TODO(getLoc(), "genval complex comparison");
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Character, KIND>> &op) {
    TODO(getLoc(), "genval char comparison");
  }

  ExtValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    TODO(getLoc(), "genval comparison");
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  ExtValue
  genval(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                          TC2> &convert) {
    TODO(getLoc(), "genval convert<TC1, KIND, TC2>");
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    TODO(getLoc(), "genval parentheses<A>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Not<KIND> &op) {
    TODO(getLoc(), "genval Not<KIND>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    TODO(getLoc(), "genval LogicalOperation<KIND>");
  }

  /// Convert a scalar literal constant to IR.
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genScalarLit(
      const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
          &value) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return genIntegerConstant<KIND>(builder.getContext(), value.ToInt64());
    } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
      return genBoolConstant(value.IsTrue());
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      TODO(getLoc(), "genval real constant");
    } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
      TODO(getLoc(), "genval complex constant");
    } else /*constexpr*/ {
      llvm_unreachable("unhandled constant");
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    if (con.Rank() > 0)
      TODO(getLoc(), "genval array constant");
    std::optional<Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>>
        opt = con.GetScalarValue();
    assert(opt.has_value() && "constant has no value");
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      TODO(getLoc(), "genval char constant");
    } else {
      return genScalarLit<TC, KIND>(opt.value());
    }
  }

  fir::ExtendedValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &con) {
    TODO(getLoc(), "genval constant derived");
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    TODO(getLoc(), "genval ArrayConstructor<A>");
  }

  ExtValue genval(const Fortran::evaluate::ComplexPart &x) {
    TODO(getLoc(), "genval ComplexPart");
  }

  ExtValue genval(const Fortran::evaluate::Substring &ss) {
    TODO(getLoc(), "genval Substring");
  }

  ExtValue genval(const Fortran::evaluate::Subscript &subs) {
    TODO(getLoc(), "genval Subscript");
  }

  ExtValue genval(const Fortran::evaluate::DataRef &dref) {
    TODO(getLoc(), "genval DataRef");
  }

  ExtValue genval(const Fortran::evaluate::Component &cmpt) {
    TODO(getLoc(), "genval Component");
  }

  ExtValue genval(const Fortran::semantics::Bound &bound) {
    TODO(getLoc(), "genval Bound");
  }

  ExtValue genval(const Fortran::evaluate::ArrayRef &aref) {
    TODO(getLoc(), "genval ArrayRef");
  }

  ExtValue genval(const Fortran::evaluate::CoarrayRef &coref) {
    TODO(getLoc(), "genval CoarrayRef");
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    TODO(getLoc(), "genval FunctionRef<A>");
  }

  ExtValue genval(const Fortran::evaluate::ProcedureRef &procRef) {
    TODO(getLoc(), "genval ProcedureRef");
  }

  template <typename A>
  bool isScalar(const A &x) {
    return x.Rank() == 0;
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x))
      return std::visit([&](const auto &e) { return genval(e); }, x.u);
    TODO(getLoc(), "genval Expr<A> arrays");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Logical, KIND>> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

private:
  mlir::Location location;
  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::SymMap &symMap;
};
} // namespace

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap}.genval(expr);
}
