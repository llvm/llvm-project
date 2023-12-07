// RUN: %clang_cc1 -fsyntax-only %s -std=c++14

// Checks that Clang doesn't crash/assert on the nested call to "kaboom"
// in "bar()".
//
// This is an interesting test case for `ExprConstant.cpp`'s `CallStackFrame`
// because it triggers the following chain of events:
// 0. `CheckEnableIf` calls `EvaluateWithSubstitution`.
//  1. The outer call to "kaboom" gets evaluated.
//   2. The expr for "a" gets evaluated, it has a version X;
//      a temporary with the key (a, X) is created.
//     3. The inner call to "kaboom" gets evaluated.
//       4. The expr for "a" gets evaluated, it has a version Y;
//          a temporary with the key (a, Y) is created.
//       5. The expr for "b" gets evaluated, it has a version Y;
//          a temporary with the key (b, Y) is created.
//   6. `EvaluateWithSubstitution` looks at "b" but cannot evaluate it
//      because it's value-dependent (due to the call to "f.foo()").
//
// When `EvaluateWithSubstitution` bails out while evaluating the outer
// call, it attempts to fetch "b"'s param slot to clean it up.
//
// This used to cause an assertion failure in `getTemporary` because
// a temporary with the key "(b, Y)" (created at step 4) existed but
// not one for "(b, X)", which is what it was trying to fetch.

template<typename T>
__attribute__((enable_if(true, "")))
T kaboom(T a, T b) {
  return b;
}

struct A {
  double foo();
};

template <int>
struct B {
  A &f;

  void bar() {
    kaboom(kaboom(0.0, 1.0), f.foo());
  }
};
