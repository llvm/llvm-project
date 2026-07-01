#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/target.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/message.h"
#include "flang/Testing/testing.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Fortran::evaluate;

int main() {
  using DefaultIntegerExpr = Expr<Type<TypeCategory::Integer>>;
  TEST(DefaultIntegerExpr::Result{subscriptIntegerKind}.AsFortran() ==
      "INTEGER(8)");
  MATCH("666_8", DefaultIntegerExpr{666}.AsFortran());
  MATCH("-1_8", (-DefaultIntegerExpr{1}).AsFortran());
  auto ex1{
      DefaultIntegerExpr{2} + DefaultIntegerExpr{3} * -DefaultIntegerExpr{4}};
  MATCH("2_8+3_8*(-4_8)", ex1.AsFortran());
  Fortran::common::IntrinsicTypeDefaultKinds defaults;
  auto intrinsics{Fortran::evaluate::IntrinsicProcTable::Configure(defaults)};
  TargetCharacteristics targetCharacteristics;
  Fortran::common::LanguageFeatureControl languageFeatures;
  std::set<std::string> tempNames;
  FoldingContext context{Fortran::parser::ContextualMessages{nullptr}, defaults,
      intrinsics, targetCharacteristics, languageFeatures, tempNames};
  ex1 = Fold(context, std::move(ex1));
  MATCH("-10_8", ex1.AsFortran());
  MATCH("1_8/2_8", (DefaultIntegerExpr{1} / DefaultIntegerExpr{2}).AsFortran());
  DefaultIntegerExpr a{1};
  DefaultIntegerExpr b{2};
  MATCH("1_8", a.AsFortran());
  a = b;
  MATCH("2_8", a.AsFortran());
  MATCH("2_8", b.AsFortran());
  return testing::Complete();
}
