// RUN: not %clang_cc1 -std=c++20 -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not='Assertion `NumElements.isPositive()` failed'

// In C++20, constexpr dynamic allocation is permitted *only* if valid.
// A negative element count must be diagnosed (and must not crash).

constexpr void f_bad_neg() {
  int a = -1;
  (void) new int[a]; // triggers negative-size path in the interpreter
}

// Force evaluation so we definitely run the constexpr interpreter.
constexpr bool force_eval = (f_bad_neg(), true);

// CHECK: error: constexpr function never produces a constant expression
