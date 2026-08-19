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

static Expr<Type<TypeCategory::Integer>> MakeDefaultIntegerExpr(int32_t v) {
  return MakeConstantExpr<Type<TypeCategory::Integer>>(
      Fortran::common::KindsEnum::Kind4, v);
}

int main() {
  using DefaultIntegerExpr = Expr<Type<TypeCategory::Integer>>;
  TEST(DefaultIntegerExpr::Result{Kind4}.AsFortran() == "INTEGER(4)");
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
  return testing::Complete();
}
