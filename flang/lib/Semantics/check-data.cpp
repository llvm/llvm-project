//===-- lib/Semantics/check-data.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DATA statement semantic analysis.
// - Applies static semantic checks to the variables in each data-stmt-set with
//   class DataVarChecker;
// - Invokes conversion of DATA statement values to static initializers

#include "check-data.h"
#include "data-to-inits.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/tools.h"
#include <algorithm>
#include <vector>

namespace Fortran::semantics {

// Ensures that references to an implied DO loop control variable are
// represented as such in the "body" of the implied DO loop.
void DataChecker::Enter(const parser::DataImpliedDo &x) {
  auto name{std::get<parser::DataImpliedDo::Bounds>(x.t).name.thing.thing};
  int kind{evaluate::ResultType<evaluate::ImpliedDoIndex>::kind};
  if (const auto dynamicType{evaluate::DynamicType::From(*name.symbol)}) {
    if (dynamicType->category() == TypeCategory::Integer) {
      kind = dynamicType->kind();
    }
  }
  exprAnalyzer_.AddImpliedDo(name.source, kind);
}

void DataChecker::Leave(const parser::DataImpliedDo &x) {
  auto name{std::get<parser::DataImpliedDo::Bounds>(x.t).name.thing.thing};
  exprAnalyzer_.RemoveImpliedDo(name.source);
}

// DataVarChecker applies static checks once to each variable that appears
// in a data-stmt-set.  These checks are independent of the values that
// correspond to the variables.
class DataVarChecker : public evaluate::AllTraverse<DataVarChecker, true> {
public:
  using Base = evaluate::AllTraverse<DataVarChecker, true>;
  DataVarChecker(SemanticsContext &c, parser::CharBlock src)
      : Base{*this}, context_{c}, source_{src} {}
  using Base::operator();
  bool HasComponentWithoutSubscripts() const {
    return hasComponent_ && !hasSubscript_;
  }
  bool operator()(const Symbol &symbol) { // C876
    // 8.6.7p(2) - precludes non-pointers of derived types with
    // default component values
    const Scope &scope{context_.FindScope(source_)};
    bool isFirstSymbol{isFirstSymbol_};
    isFirstSymbol_ = false;
    // Ordered so that most egregious errors are first
    if (const char *whyNot{IsProcedure(symbol) && !IsPointer(symbol)
                ? "Procedure"
                : isFirstSymbol && IsHostAssociated(symbol, scope)
                ? "Host-associated object"
                : isFirstSymbol && IsUseAssociated(symbol, scope)
                ? "USE-associated object"
                : IsDummy(symbol)          ? "Dummy argument"
                : IsFunctionResult(symbol) ? "Function result"
                : IsAutomatic(symbol)      ? "Automatic variable"
                : IsAllocatable(symbol)    ? "Allocatable"
                : IsInitialized(symbol, true /*ignore DATA*/,
                      true /*ignore allocatable components*/,
                      true /*ignore uninitialized pointer components*/)
                ? "Default-initialized"
                : symbol.has<AssocEntityDetails>() ? "Construct association"
                : isFirstSymbol && IsPointer(symbol) &&
                    (hasComponent_ || hasSubscript_)
                ? "Target of pointer"
                : nullptr}) {
      context_.Say(source_,
          "%s '%s' must not be initialized in a DATA statement"_err_en_US,
          whyNot, symbol.name());
      return false;
    }
    if (IsProcedurePointer(symbol)) {
      if (!context_.IsEnabled(common::LanguageFeature::DataStmtExtensions)) {
        context_.Say(source_,
            "Procedure pointer '%s' may not appear in a DATA statement"_err_en_US,
            symbol.name());
        return false;
      } else if (context_.ShouldWarn(
                     common::LanguageFeature::DataStmtExtensions)) {
        context_.Say(source_,
            "Procedure pointer '%s' in a DATA statement is not standard"_port_en_US,
            symbol.name());
      }
    }
    if (IsInBlankCommon(symbol)) {
      if (!context_.IsEnabled(common::LanguageFeature::DataStmtExtensions)) {
        context_.Say(source_,
            "Blank COMMON object '%s' may not appear in a DATA statement"_err_en_US,
            symbol.name());
        return false;
      } else if (context_.ShouldWarn(
                     common::LanguageFeature::DataStmtExtensions)) {
        context_.Say(source_,
            "Blank COMMON object '%s' in a DATA statement is not standard"_port_en_US,
            symbol.name());
      }
    }
    return true;
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
      auto restorer{common::ScopedSet(isPointerAllowed_, false)};
      return (*this)(component.base()) && (*this)(lastSymbol);
    } else if (IsPointer(lastSymbol)) { // C877
      context_.Say(source_,
          "Data object must not contain pointer '%s' as a non-rightmost part"_err_en_US,
          lastSymbol.name().ToString());
      return false;
    } else {
      return (*this)(component.base()) && (*this)(lastSymbol);
    }
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
    context_.Say(
        source_, "Data object must not be a coindexed variable"_err_en_US);
    return false;
  }
  bool operator()(const evaluate::Subscript &subs) {
    auto restorer1{common::ScopedSet(isPointerAllowed_, false)};
    auto restorer2{common::ScopedSet(isFunctionAllowed_, true)};
    return common::visit(
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
    if (isFunctionAllowed_) {
      // Must have been validated as a constant expression
      return true;
    } else {
      context_.Say(source_,
          "Data object variable must not be a function reference"_err_en_US);
      return false;
    }
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
  bool isPointerAllowed_{true};
  bool isFirstSymbol_{true};
  bool isFunctionAllowed_{false};
};

static bool IsValidDataObject(const SomeExpr &expr) { // C878, C879
  return !evaluate::IsConstantExpr(expr) &&
      (evaluate::IsVariable(expr) || evaluate::IsProcedurePointer(expr));
}

void DataChecker::Leave(const parser::DataIDoObject &object) {
  if (const auto *designator{
          std::get_if<parser::Scalar<common::Indirection<parser::Designator>>>(
              &object.u)}) {
    if (MaybeExpr expr{exprAnalyzer_.Analyze(*designator)}) {
      auto source{designator->thing.value().source};
      DataVarChecker checker{exprAnalyzer_.context(), source};
      if (checker(*expr)) {
        if (checker.HasComponentWithoutSubscripts()) { // C880
          exprAnalyzer_.context().Say(source,
              "Data implied do structure component must be subscripted"_err_en_US);
        } else if (!IsValidDataObject(*expr)) {
          exprAnalyzer_.context().Say(
              source, "Data implied do object must be a variable"_err_en_US);
        } else {
          return;
        }
      }
    }
    currentSetHasFatalErrors_ = true;
  }
}

void DataChecker::Leave(const parser::DataStmtObject &dataObject) {
  common::visit(
      common::visitors{
          [](const parser::DataImpliedDo &) { // has own Enter()/Leave()
          },
          [&](const auto &var) {
            auto expr{exprAnalyzer_.Analyze(var)};
            auto source{parser::FindSourceLocation(dataObject)};
            if (!expr ||
                !DataVarChecker{exprAnalyzer_.context(), source}(*expr)) {
              currentSetHasFatalErrors_ = true;
            } else if (!IsValidDataObject(*expr)) {
              exprAnalyzer_.context().Say(
                  source, "Data statement object must be a variable"_err_en_US);
              currentSetHasFatalErrors_ = true;
            }
          },
      },
      dataObject.u);
}

void DataChecker::Leave(const parser::DataStmtSet &set) {
  if (!currentSetHasFatalErrors_) {
    AccumulateDataInitializations(inits_, exprAnalyzer_, set);
  }
  currentSetHasFatalErrors_ = false;
}

// Handle legacy DATA-style initialization, e.g. REAL PI/3.14159/, for
// variables and components (esp. for DEC STRUCTUREs)
template <typename A> void DataChecker::LegacyDataInit(const A &decl) {
  if (const auto &init{
          std::get<std::optional<parser::Initialization>>(decl.t)}) {
    const Symbol *name{std::get<parser::Name>(decl.t).symbol};
    const auto *list{
        std::get_if<std::list<common::Indirection<parser::DataStmtValue>>>(
            &init->u)};
    if (name && list) {
      AccumulateDataInitializations(inits_, exprAnalyzer_, *name, *list);
    }
  }
}

void DataChecker::Leave(const parser::ComponentDecl &decl) {
  LegacyDataInit(decl);
}

void DataChecker::Leave(const parser::EntityDecl &decl) {
  LegacyDataInit(decl);
}

void DataChecker::CompileDataInitializationsIntoInitializers() {
  ConvertToInitializers(inits_, exprAnalyzer_);
}

} // namespace Fortran::semantics
