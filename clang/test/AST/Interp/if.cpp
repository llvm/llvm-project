// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fexperimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1 -std=c++23 -fsyntax-only %s -verify=ref

// expected-no-diagnostics
// ref-no-diagnostics

namespace ConstEval {
  constexpr int f() {
    int i = 0;
    if consteval {
      i = 1;
    }
    return i;
  }
  static_assert(f() == 1, "");

  constexpr int f2() {
    int i = 0;
    if !consteval {
        i = 12;
      if consteval {
        i = i + 1;
      }
    }
    return i;
  }
  static_assert(f2() == 0, "");
};

namespace InitDecl {
  constexpr bool f() {
    if (int i = 5; i != 10) {
      return true;
    }
    return false;
  }
  static_assert(f(), "");

  constexpr bool f2() {
    if (bool b = false; b) {
      return true;
    }
    return false;
  }
  static_assert(!f2(), "");
};
