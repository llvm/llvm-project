//===-- ConvertExprToHLFIR.cpp --------------------------------------------===//
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

#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/ConvertConstant.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Todo.h"

namespace {

/// Lower Designators to HLFIR.
class HlfirDesignatorBuilder {
public:
  HlfirDesignatorBuilder(mlir::Location loc,
                         Fortran::lower::AbstractConverter &converter,
                         Fortran::lower::SymMap &symMap,
                         Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, symMap{symMap}, stmtCtx{stmtCtx}, loc{loc} {}

  // Character designators variant contains substrings
  using CharacterDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Character, 1>>::u);
  hlfir::EntityWithAttributes
  gen(const CharacterDesignators &designatorVariant) {
    return std::visit([&](const auto &x) { return gen(x); }, designatorVariant);
  }
  // Character designators variant contains complex parts
  using RealDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Real, 4>>::u);
  hlfir::EntityWithAttributes gen(const RealDesignators &designatorVariant) {
    return std::visit([&](const auto &x) { return gen(x); }, designatorVariant);
  }
  // All other designators are similar
  using OtherDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Integer, 4>>::u);
  hlfir::EntityWithAttributes gen(const OtherDesignators &designatorVariant) {
    return std::visit([&](const auto &x) { return gen(x); }, designatorVariant);
  }

private:
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::SymbolRef &symbolRef) {
    if (llvm::Optional<fir::FortranVariableOpInterface> varDef =
            getSymMap().lookupVariableDefinition(symbolRef))
      return *varDef;
    TODO(getLoc(), "lowering symbol to HLFIR");
  }
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Component &component) {
    TODO(getLoc(), "lowering component to HLFIR");
  }
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::ArrayRef &arrayRef) {
    TODO(getLoc(), "lowering ArrayRef to HLFIR");
  }
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::CoarrayRef &coarrayRef) {
    TODO(getLoc(), "lowering CoarrayRef to HLFIR");
  }
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ComplexPart &complexPart) {
    TODO(getLoc(), "lowering complex part to HLFIR");
  }
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Substring &substring) {
    TODO(getLoc(), "lowering substrings to HLFIR");
  }

  mlir::Location getLoc() const { return loc; }
  Fortran::lower::AbstractConverter &getConverter() { return converter; }
  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }
  Fortran::lower::SymMap &getSymMap() { return symMap; }
  Fortran::lower::StatementContext &getStmtCtx() { return stmtCtx; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  mlir::Location loc;
};

/// Lower Expr to HLFIR.
class HlfirBuilder {
public:
  HlfirBuilder(mlir::Location loc, Fortran::lower::AbstractConverter &converter,
               Fortran::lower::SymMap &symMap,
               Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, symMap{symMap}, stmtCtx{stmtCtx}, loc{loc} {}

  template <typename T>
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::Expr<T> &expr) {
    return std::visit([&](const auto &x) { return gen(x); }, expr.u);
  }

private:
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::BOZLiteralConstant &expr) {
    fir::emitFatalError(loc, "BOZ literal must be replaced by semantics");
  }
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::NullPointer &expr) {
    TODO(getLoc(), "lowering NullPointer to HLFIR");
  }
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ProcedureDesignator &expr) {
    TODO(getLoc(), "lowering ProcDes to HLFIR");
  }
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::ProcedureRef &expr) {
    TODO(getLoc(), "lowering ProcRef to HLFIR");
  }

  template <typename T>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Designator<T> &designator) {
    return HlfirDesignatorBuilder(getLoc(), getConverter(), getSymMap(),
                                  getStmtCtx())
        .gen(designator.u);
  }

  template <typename T>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::FunctionRef<T> &expr) {
    TODO(getLoc(), "lowering funcRef to HLFIR");
  }

  template <typename T>
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::Constant<T> &expr) {
    mlir::Location loc = getLoc();
    if constexpr (std::is_same_v<T, Fortran::evaluate::SomeDerived>) {
      TODO(loc, "lowering derived type constant to HLFIR");
    } else {
      fir::FirOpBuilder &builder = getBuilder();
      fir::ExtendedValue exv =
          Fortran::lower::IntrinsicConstantBuilder<T::category, T::kind>::gen(
              builder, loc, expr, /*outlineBigConstantInReadOnlyMemory=*/true);
      if (const auto *scalarBox = exv.getUnboxed())
        if (fir::isa_trivial(scalarBox->getType()))
          return hlfir::EntityWithAttributes(*scalarBox);
      if (auto addressOf = fir::getBase(exv).getDefiningOp<fir::AddrOfOp>()) {
        auto flags = fir::FortranVariableFlagsAttr::get(
            builder.getContext(), fir::FortranVariableFlagsEnum::parameter);
        return hlfir::genDeclare(
            loc, builder, exv,
            addressOf.getSymbol().getRootReference().getValue(), flags);
      }
      fir::emitFatalError(loc, "Constant<T> was lowered to unexpected format");
    }
  }

  template <typename T>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ArrayConstructor<T> &expr) {
    TODO(getLoc(), "lowering ArrayCtor to HLFIR");
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>, TC2>
          &convert) {
    TODO(getLoc(), "lowering convert to HLFIR");
  }

  template <typename D, typename R, typename O>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Operation<D, R, O> &op) {
    TODO(getLoc(), "lowering unary op to HLFIR");
  }

  template <typename D, typename R, typename LO, typename RO>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Operation<D, R, LO, RO> &op) {
    TODO(getLoc(), "lowering binary op to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return gen(x); }, op.u);
  }

  hlfir::EntityWithAttributes gen(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "lowering type parameter inquiry to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::DescriptorInquiry &desc) {
    TODO(getLoc(), "lowering descriptor inquiry to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ImpliedDoIndex &var) {
    TODO(getLoc(), "lowering implied do index to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::StructureConstructor &var) {
    TODO(getLoc(), "lowering structure constructor to HLFIR");
  }

  mlir::Location getLoc() const { return loc; }
  Fortran::lower::AbstractConverter &getConverter() { return converter; }
  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }
  Fortran::lower::SymMap &getSymMap() { return symMap; }
  Fortran::lower::StatementContext &getStmtCtx() { return stmtCtx; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  mlir::Location loc;
};

} // namespace

hlfir::EntityWithAttributes Fortran::lower::convertExprToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  return HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
}
