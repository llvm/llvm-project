#include "flang/Evaluate/expression.h"
#include "testing.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/target.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/message.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Fortran::evaluate;

int main() {
  using DefaultIntegerExpr = Expr<Type<TypeCategory::Integer, 4>>;
  TEST(DefaultIntegerExpr::Result::AsFortran() == "INTEGER(4)");
  MATCH("666_4", DefaultIntegerExpr{666}.AsFortran());
  MATCH("-1_4", (-DefaultIntegerExpr{1}).AsFortran());
  auto ex1{
      DefaultIntegerExpr{2} + DefaultIntegerExpr{3} * -DefaultIntegerExpr{4}};
  MATCH("2_4+3_4*(-4_4)", ex1.AsFortran());
  Fortran::common::IntrinsicTypeDefaultKinds defaults;
  auto intrinsics{Fortran::evaluate::IntrinsicProcTable::Configure(defaults)};
  TargetCharacteristics targetCharacteristics;
  FoldingContext context{Fortran::parser::ContextualMessages{nullptr}, defaults,
      intrinsics, targetCharacteristics};
  ex1 = Fold(context, std::move(ex1));
  MATCH("-10_4", ex1.AsFortran());
  MATCH("1_4/2_4", (DefaultIntegerExpr{1} / DefaultIntegerExpr{2}).AsFortran());
  DefaultIntegerExpr a{1};
  DefaultIntegerExpr b{2};
  MATCH("1_4", a.AsFortran());
  a = b;
  MATCH("2_4", a.AsFortran());
  MATCH("2_4", b.AsFortran());
  return testing::Complete();
}
