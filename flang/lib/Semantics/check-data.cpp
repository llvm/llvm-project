//===-- lib/Semantics/check-data.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-data.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Semantics/expression.h"

namespace Fortran::semantics {

void DataChecker::Leave(const parser::DataStmtConstant &dataConst) {
  if (auto *structure{
          std::get_if<parser::StructureConstructor>(&dataConst.u)}) {
    for (const auto &component :
        std::get<std::list<parser::ComponentSpec>>(structure->t)) {
      const parser::Expr &parsedExpr{
          std::get<parser::ComponentDataSource>(component.t).v.value()};
      if (const auto *expr{GetExpr(parsedExpr)}) {
        if (!evaluate::IsConstantExpr(*expr)) { // C884
          exprAnalyzer_.Say(parsedExpr.source,
              "Structure constructor in data value must be a constant expression"_err_en_US);
        }
      }
    }
  }
}

// Ensures that references to an implied DO loop control variable are
// represented as such in the "body" of the implied DO loop.
void DataChecker::Enter(const parser::DataImpliedDo &x) {
  auto name{std::get<parser::DataImpliedDo::Bounds>(x.t).name.thing.thing};
  int kind{evaluate::ResultType<evaluate::ImpliedDoIndex>::kind};
  if (const auto dynamicType{evaluate::DynamicType::From(*name.symbol)}) {
    kind = dynamicType->kind();
  }
  exprAnalyzer_.AddImpliedDo(name.source, kind);
}

void DataChecker::Leave(const parser::DataImpliedDo &x) {
  auto name{std::get<parser::DataImpliedDo::Bounds>(x.t).name.thing.thing};
  exprAnalyzer_.RemoveImpliedDo(name.source);
}

class DataVarChecker : public evaluate::AllTraverse<DataVarChecker, true> {
public:
  using Base = evaluate::AllTraverse<DataVarChecker, true>;
  DataVarChecker(SemanticsContext &c, parser::CharBlock src)
      : Base{*this}, context_{c}, source_{src} {}
  using Base::operator();
  bool HasComponentWithoutSubscripts() const {
    return hasComponent_ && !hasSubscript_;
  }
  bool operator()(const evaluate::Component &component) {
    hasComponent_ = true;
    return (*this)(component.base());
  }
  bool operator()(const evaluate::Subscript &subs) {
    hasSubscript_ = true;
    return std::visit(
        common::visitors{
            [&](const evaluate::IndirectSubscriptIntegerExpr &expr) {
              return CheckSubscriptExpr(expr);
            },
            [&](const evaluate::Triplet &triplet) {
              return CheckSubscriptExpr(triplet.lower()) &&
                  CheckSubscriptExpr(triplet.upper()) &&
                  CheckSubscriptExpr(triplet.stride());
            },
        },
        subs.u);
  }
  template <typename T>
  bool operator()(const evaluate::FunctionRef<T> &) const { // C875
    context_.Say(source_,
        "Data object variable must not be a function reference"_err_en_US);
    return false;
  }
  bool operator()(const evaluate::CoarrayRef &) const { // C874
    context_.Say(
        source_, "Data object must not be a coindexed variable"_err_en_US);
    return false;
  }

private:
  bool CheckSubscriptExpr(
      const std::optional<evaluate::IndirectSubscriptIntegerExpr> &x) const {
    return !x || CheckSubscriptExpr(*x);
  }
  bool CheckSubscriptExpr(
      const evaluate::IndirectSubscriptIntegerExpr &expr) const {
    return CheckSubscriptExpr(expr.value());
  }
  bool CheckSubscriptExpr(
      const evaluate::Expr<evaluate::SubscriptInteger> &expr) const {
    if (!evaluate::IsConstantExpr(expr)) { // C875,C881
      context_.Say(
          source_, "Data object must have constant subscripts"_err_en_US);
      return false;
    } else {
      return true;
    }
  }

  SemanticsContext &context_;
  parser::CharBlock source_;
  bool hasComponent_{false};
  bool hasSubscript_{false};
};

// TODO: C876, C877, C879
void DataChecker::Leave(const parser::DataIDoObject &object) {
  if (const auto *designator{
          std::get_if<parser::Scalar<common::Indirection<parser::Designator>>>(
              &object.u)}) {
    if (MaybeExpr expr{exprAnalyzer_.Analyze(*designator)}) {
      auto source{designator->thing.value().source};
      if (evaluate::IsConstantExpr(*expr)) { // C878
        exprAnalyzer_.Say(
            source, "Data implied do object must be a variable"_err_en_US);
      } else {
        DataVarChecker checker{exprAnalyzer_.context(), source};
        if (checker(*expr) && checker.HasComponentWithoutSubscripts()) { // C880
          exprAnalyzer_.Say(source,
              "Data implied do structure component must be subscripted"_err_en_US);
        }
      }
    }
  }
}

void DataChecker::Leave(const parser::DataStmtObject &dataObject) {
  if (const auto *var{
          std::get_if<common::Indirection<parser::Variable>>(&dataObject.u)}) {
    if (auto expr{exprAnalyzer_.Analyze(*var)}) {
      DataVarChecker{exprAnalyzer_.context(),
          parser::FindSourceLocation(dataObject)}(expr);
    }
  }
}

void DataChecker::Leave(const parser::DataStmtRepeat &dataRepeat) {
  if (const auto *designator{parser::Unwrap<parser::Designator>(dataRepeat)}) {
    if (auto *dataRef{std::get_if<parser::DataRef>(&designator->u)}) {
      if (MaybeExpr checked{exprAnalyzer_.Analyze(*dataRef)}) {
        auto expr{evaluate::Fold(
            exprAnalyzer_.GetFoldingContext(), std::move(checked))};
        if (auto i64{ToInt64(expr)}) {
          if (*i64 < 0) { // C882
            exprAnalyzer_.Say(designator->source,
                "Repeat count for data value must not be negative"_err_en_US);
          }
        }
      }
    }
  }
}
} // namespace Fortran::semantics
