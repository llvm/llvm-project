// RUN: %clang_cc1 -std=c++1y -o - -emit-llvm -verify %s
// RUN: %clang_cc1 -std=c++1y -fexperimental-new-constant-interpreter -o - -emit-llvm -verify %s

namespace default_arg_temporary {

constexpr bool equals(const float& arg = 1.0f) {
  return arg == 1.0f;
}

constexpr const int &x(const int &p = 0) {
  return p;
}

struct S {
  constexpr S(const int &a = 0) {}
};

void test_default_arg2() {
  // This piece of code used to cause an assertion failure in
  // CallStackFrame::createTemporary because the same MTE is used to initilize
  // both elements of the array (see PR33140).
  constexpr S s[2] = {};

  // This piece of code used to cause an assertion failure in
  // CallStackFrame::createTemporary because multiple CXXDefaultArgExpr share
  // the same MTE (see PR33140).
  static_assert(equals() && equals(), "");

  // Test that constant expression evaluation produces distinct lvalues for
  // each call.
  static_assert(&x() != &x(), "");
}

// Check that multiple CXXDefaultInitExprs don't cause an assertion failure.
struct A { int &&r = 0; };
struct B { A x, y; };
B b = {}; // expected-no-diagnostics

}
