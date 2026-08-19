#include "flang/Evaluate/type.h"
#include "flang/Testing/testing.h"
#include <cstdio>

static void testKind(int kind) {
  using Type = Fortran::evaluate::Type<Fortran::common::TypeCategory::Logical>;
  TEST(Fortran::evaluate::IsSpecificIntrinsicType<Type>);
  TEST(Type::category == Fortran::common::TypeCategory::Logical);
  TEST(Type{kind}.kind() == kind);
  using Value = Fortran::evaluate::Scalar<Type>;
  MATCH(8 * kind, Value::Zero(kind).bits());
  TEST(!Value{}.IsTrue());
  TEST(!Value(kind, false).IsTrue());
  TEST(Value(kind, true).IsTrue());
  TEST(Value(kind, false).NOT().IsTrue());
  TEST(!Value(kind, true).NOT().IsTrue());
  TEST(!Value(kind, false).AND(Value(kind, false)).IsTrue());
  TEST(!Value(kind, false).AND(Value(kind, true)).IsTrue());
  TEST(!Value(kind, true).AND(Value(kind, false)).IsTrue());
  TEST(Value(kind, true).AND(Value(kind, true)).IsTrue());
  TEST(!Value(kind, false).OR(Value(kind, false)).IsTrue());
  TEST(Value(kind, false).OR(Value(kind, true)).IsTrue());
  TEST(Value(kind, true).OR(Value(kind, false)).IsTrue());
  TEST(Value(kind, true).OR(Value(kind, true)).IsTrue());
  TEST(Value(kind, false).EQV(Value(kind, false)).IsTrue());
  TEST(!Value(kind, false).EQV(Value(kind, true)).IsTrue());
  TEST(!Value(kind, true).EQV(Value(kind, false)).IsTrue());
  TEST(Value(kind, true).EQV(Value(kind, true)).IsTrue());
  TEST(!Value(kind, false).NEQV(Value(kind, false)).IsTrue());
  TEST(Value(kind, false).NEQV(Value(kind, true)).IsTrue());
  TEST(Value(kind, true).NEQV(Value(kind, false)).IsTrue());
  TEST(!Value(kind, true).NEQV(Value(kind, true)).IsTrue());
}

int main() {
  testKind(1);
  testKind(2);
  testKind(4);
  testKind(8);
  return testing::Complete();
}
