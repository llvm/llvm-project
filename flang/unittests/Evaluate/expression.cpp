//===-- unittests/Evaluate/expression.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/target.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/message.h"
#include "flang/Parser/provenance.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Testing/testing.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Fortran::evaluate;

static Expr<Type<TypeCategory::Integer>> MakeDefaultIntegerExpr(int32_t v) {
  return MakeConstantExpr<Type<TypeCategory::Integer>>(4, v);
}

int main() {
  using DefaultIntegerExpr = Expr<Type<TypeCategory::Integer>>;
  const int defaultIntegerKind{4};
  TEST(DefaultIntegerExpr::Result{4}.AsFortran() == "INTEGER(4)");
  MATCH("666_4", MakeDefaultIntegerExpr(666).AsFortran());
  MATCH("-1_4", (-MakeDefaultIntegerExpr(1)).AsFortran());
  auto ex1{MakeDefaultIntegerExpr(2) +
      MakeDefaultIntegerExpr(3) * -MakeDefaultIntegerExpr(4)};
  MATCH("2_4+3_4*(-4_4)", ex1.AsFortran());
  Fortran::common::IntrinsicTypeDefaultKinds defaults;
  auto intrinsics{Fortran::evaluate::IntrinsicProcTable::Configure(defaults)};
  TargetCharacteristics targetCharacteristics;
  Fortran::common::LanguageFeatureControl languageFeatures;
  std::set<std::string> tempNames;
  FoldingContext context{Fortran::parser::ContextualMessages{nullptr}, defaults,
      intrinsics, targetCharacteristics, languageFeatures, tempNames};
  ex1 = Fold(context, std::move(ex1));
  MATCH("-10_4", ex1.AsFortran());
  MATCH("1_4/2_4",
      (MakeDefaultIntegerExpr(1) / MakeDefaultIntegerExpr(2)).AsFortran());
  DefaultIntegerExpr a{MakeDefaultIntegerExpr(1)};
  DefaultIntegerExpr b{MakeDefaultIntegerExpr(2)};
  MATCH("1_4", a.AsFortran());
  a = b;
  MATCH("2_4", a.AsFortran());
  MATCH("2_4", b.AsFortran());

  Fortran::parser::AllSources allSources;
  Fortran::parser::AllCookedSources cookedSources{allSources};
  Fortran::common::LangOptions langOptions;
  Fortran::semantics::SemanticsContext semantics{
      defaults, languageFeatures, langOptions, cookedSources};
  auto &scope{semantics.globalScope()};
  const auto &integerType{
      scope.MakeNumericType(TypeCategory::Integer, MakeDefaultIntegerExpr(4))};
  auto &coarray{scope.MakeSymbol(Fortran::parser::CharBlock{"a", 1}, {},
      Fortran::semantics::ObjectEntityDetails{})};
  auto &status{scope.MakeSymbol(Fortran::parser::CharBlock{"s", 1}, {},
      Fortran::semantics::ObjectEntityDetails{})};
  coarray.get<Fortran::semantics::ObjectEntityDetails>().set_type(integerType);
  status.get<Fortran::semantics::ObjectEntityDetails>().set_type(integerType);
  auto asExpr{[](const CoarrayRef &ref) {
    return DefaultIntegerExpr{Designator<DefaultIntegerExpr::Result>{
        defaultIntegerKind, DataRef{ref}}};
  }};

  CoarrayRef noNotify{
      DataRef{coarray}, {MakeSubscriptIntExpr(1) + MakeSubscriptIntExpr(1)}};
  CoarrayRef withNotify{noNotify};
  // Integer expressions exercise the type-independent specifier representation.
  withNotify.set_notify(
      AsGenericExpr(MakeDefaultIntegerExpr(2) + MakeDefaultIntegerExpr(3)));
  MATCH("a[1_8+1_8,NOTIFY=2_4+3_4]", asExpr(withNotify).AsFortran());

  auto foldedNotify{ExtractCoarrayRef(Fold(context, asExpr(withNotify)))};
  TEST(foldedNotify.has_value());
  if (foldedNotify) {
    TEST(foldedNotify->notify().has_value());
    MATCH("2_8", foldedNotify->cosubscript().front().AsFortran());
    MATCH("a[2_8,NOTIFY=5_4]", asExpr(*foldedNotify).AsFortran());
    if (auto notify{foldedNotify->notify()}) {
      MATCH("5_4", notify->AsFortran());
    }
  }
  auto foldedAbsent{ExtractCoarrayRef(Fold(context, asExpr(noNotify)))};
  TEST(foldedAbsent.has_value());
  if (foldedAbsent) {
    TEST(!foldedAbsent->notify().has_value());
    MATCH("a[2_8]", asExpr(*foldedAbsent).AsFortran());
  }

  CoarrayRef combined{withNotify};
  combined.set_stat(
      AsCategoryExpr(DefaultIntegerExpr{Designator<DefaultIntegerExpr::Result>{
          defaultIntegerKind, DataRef{status}}}));
  combined.set_team(
      AsGenericExpr(MakeDefaultIntegerExpr(7) + MakeDefaultIntegerExpr(1)));
  auto foldedCombined{ExtractCoarrayRef(Fold(context, asExpr(combined)))};
  TEST(foldedCombined.has_value());
  if (foldedCombined) {
    TEST(foldedCombined->notify().has_value());
    TEST(foldedCombined->stat().has_value());
    TEST(foldedCombined->team().has_value());
    MATCH("a[2_8,STAT=s,NOTIFY=5_4,TEAM_NUMBER=8_4]",
        asExpr(*foldedCombined).AsFortran());
    if (auto stat{foldedCombined->stat()}) {
      MATCH("s", stat->AsFortran());
    }
    if (auto team{foldedCombined->team()}) {
      MATCH("8_4", team->AsFortran());
    }
  }
  return testing::Complete();
}
