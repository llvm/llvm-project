#include "testing.h"
#include "flang/Evaluate/type.h"
#include <cstdio>

template <int KIND> void testKind() {
  using Type =
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Logical, KIND>;
  TEST(Fortran::evaluate::IsSpecificIntrinsicType<Type>);
  TEST(Type::category == Fortran::common::TypeCategory::Logical);
  TEST(Type::kind == KIND);
  using Value = Fortran::evaluate::Scalar<Type>;
  MATCH(8 * KIND, Value::bits);
  TEST(!Value{}.IsTrue());
  TEST(!Value{false}.IsTrue());
  TEST(Value{true}.IsTrue());
  TEST(Value{false}.NOT().IsTrue());
  TEST(!Value{true}.NOT().IsTrue());
  TEST(!Value{false}.AND(Value{false}).IsTrue());
  TEST(!Value{false}.AND(Value{true}).IsTrue());
  TEST(!Value{true}.AND(Value{false}).IsTrue());
  TEST(Value{true}.AND(Value{true}).IsTrue());
  TEST(!Value{false}.OR(Value{false}).IsTrue());
  TEST(Value{false}.OR(Value{true}).IsTrue());
  TEST(Value{true}.OR(Value{false}).IsTrue());
  TEST(Value{true}.OR(Value{true}).IsTrue());
  TEST(Value{false}.EQV(Value{false}).IsTrue());
  TEST(!Value{false}.EQV(Value{true}).IsTrue());
  TEST(!Value{true}.EQV(Value{false}).IsTrue());
  TEST(Value{true}.EQV(Value{true}).IsTrue());
  TEST(!Value{false}.NEQV(Value{false}).IsTrue());
  TEST(Value{false}.NEQV(Value{true}).IsTrue());
  TEST(Value{true}.NEQV(Value{false}).IsTrue());
  TEST(!Value{true}.NEQV(Value{true}).IsTrue());
}

int main() {
  testKind<1>();
  testKind<2>();
  testKind<4>();
  testKind<8>();
  return testing::Complete();
}
