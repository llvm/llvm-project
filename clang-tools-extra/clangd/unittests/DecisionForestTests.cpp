#include "DecisionForestRuntimeTest.h"
#include "decision_forest_model/CategoricalFeature.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

TEST(DecisionForestRuntime, Evaluate) {
  using Example = ::ns1::ns2::test::Example;
  using Cat = ::ns1::ns2::TestEnum;
  using ::ns1::ns2::test::Evaluate;

  Example E;
  E.setANumber(200);         // True
  E.setAFloat(0);            // True: +10.0
  E.setACategorical(Cat::A); // True: +5.0
  EXPECT_EQ(Evaluate(E), 15.0);

  E.setANumber(200);         // True
  E.setAFloat(-2.5);         // False: -20.0
  E.setACategorical(Cat::B); // True: +5.0
  EXPECT_EQ(Evaluate(E), -15.0);

  E.setANumber(100);         // False
  E.setACategorical(Cat::C); // True: +3.0, False: -6.0
  EXPECT_EQ(Evaluate(E), -3.0);
}
} // namespace clangd
} // namespace clang
