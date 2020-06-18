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
    const Symbol &lastSymbol{component.GetLastSymbol()};
    if (isPointerAllowed_) {
      if (IsPointer(lastSymbol) && hasSubscript_) { // C877
        context_.Say(source_,
            "Rightmost data object pointer '%s' must not be subscripted"_err_en_US,
            lastSymbol.name().ToString());
        return false;
      }
      RestrictPointer();
    } else {
      if (IsPointer(lastSymbol)) { // C877
        context_.Say(source_,
            "Data object must not contain pointer '%s' as a non-rightmost part"_err_en_US,
            lastSymbol.name().ToString());
        return false;
      }
    }
    if (!isFirstSymbolChecked_) {
      isFirstSymbolChecked_ = true;
      if (!CheckFirstSymbol(component.GetFirstSymbol())) {
        return false;
      }
    }
    return (*this)(component.base()) && (*this)(lastSymbol);
  }
  bool operator()(const evaluate::ArrayRef &arrayRef) {
    hasSubscript_ = true;
    return (*this)(arrayRef.base()) && (*this)(arrayRef.subscript());
  }
  bool operator()(const evaluate::Substring &substring) {
    hasSubscript_ = true;
    return (*this)(substring.parent()) && (*this)(substring.lower()) &&
        (*this)(substring.upper());
  }
  bool operator()(const evaluate::CoarrayRef &) { // C874
    hasSubscript_ = true;
    context_.Say(
        source_, "Data object must not be a coindexed variable"_err_en_US);
    return false;
  }
  bool operator()(const evaluate::Symbol &symbol) {
    if (!isFirstSymbolChecked_) {
      return CheckFirstSymbol(symbol) && CheckAnySymbol(symbol);
    } else {
      return CheckAnySymbol(symbol);
    }
  }
  bool operator()(const evaluate::Subscript &subs) {
    DataVarChecker subscriptChecker{context_, source_};
    subscriptChecker.RestrictPointer();
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
               subs.u) &&
        subscriptChecker(subs.u);
  }
  template <typename T>
  bool operator()(const evaluate::FunctionRef<T> &) const { // C875
    context_.Say(source_,
        "Data object variable must not be a function reference"_err_en_US);
    return false;
  }
  void RestrictPointer() { isPointerAllowed_ = false; }

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
  bool CheckFirstSymbol(const Symbol &symbol);
  bool CheckAnySymbol(const Symbol &symbol);

  SemanticsContext &context_;
  parser::CharBlock source_;
  bool hasComponent_{false};
  bool hasSubscript_{false};
  bool isPointerAllowed_{true};
  bool isFirstSymbolChecked_{false};
};

bool DataVarChecker::CheckFirstSymbol(const Symbol &symbol) { // C876
  const Scope &scope{context_.FindScope(source_)};
  if (IsDummy(symbol)) {
    context_.Say(source_,
        "Data object part '%s' must not be a dummy argument"_err_en_US,
        symbol.name().ToString());
  } else if (IsFunction(symbol)) {
    context_.Say(source_,
        "Data object part '%s' must not be a function name"_err_en_US,
        symbol.name().ToString());
  } else if (symbol.IsFuncResult()) {
    context_.Say(source_,
        "Data object part '%s' must not be a function result"_err_en_US,
        symbol.name().ToString());
  } else if (IsHostAssociated(symbol, scope)) {
    context_.Say(source_,
        "Data object part '%s' must not be accessed by host association"_err_en_US,
        symbol.name().ToString());
  } else if (IsUseAssociated(symbol, scope)) {
    context_.Say(source_,
        "Data object part '%s' must not be accessed by use association"_err_en_US,
        symbol.name().ToString());
  } else if (IsInBlankCommon(symbol)) {
    context_.Say(source_,
        "Data object part '%s' must not be in blank COMMON"_err_en_US,
        symbol.name().ToString());
  } else {
    return true;
  }
  return false;
}

bool DataVarChecker::CheckAnySymbol(const Symbol &symbol) { // C876
  if (IsAutomaticObject(symbol)) {
    context_.Say(source_,
        "Data object part '%s' must not be an automatic object"_err_en_US,
        symbol.name().ToString());
  } else if (IsAllocatable(symbol)) {
    context_.Say(source_,
        "Data object part '%s' must not be an allocatable object"_err_en_US,
        symbol.name().ToString());
  } else {
    return true;
  }
  return false;
}

void DataChecker::Leave(const parser::DataIDoObject &object) {
  if (const auto *designator{
          std::get_if<parser::Scalar<common::Indirection<parser::Designator>>>(
              &object.u)}) {
    if (MaybeExpr expr{exprAnalyzer_.Analyze(*designator)}) {
      auto source{designator->thing.value().source};
      if (evaluate::IsConstantExpr(*expr)) { // C878,C879
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
