// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -fexperimental-new-constant-interpreter
// expected-no-diagnostics

/*
FIXME: Support constexpr

constexpr int evaluate_once(int x) {
  return (++x) ? : 10;
}
static_assert(evaluate_once(0) == 1, "");
*/

namespace GH15998 {
  enum E { Zero, One };
  E test(E e) {
    return e ? : One;
  }
}
