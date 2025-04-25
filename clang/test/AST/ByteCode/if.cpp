// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fexperimental-new-constant-interpreter %s -verify=expected,both
// RUN: %clang_cc1 -std=c++23 -fsyntax-only %s -verify=ref,both

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


  constexpr int attrs() {
    if (1) [[likely]] {}
    return 1;
  }
  static_assert(attrs() == 1, "");
};

/// The faulty if statement creates a RecoveryExpr with contains-errors,
/// but the execution will never reach that.
constexpr char g(char const (&x)[2]) {
    return 'x';
  if (auto [a, b] = x) // both-error {{an array type is not allowed here}} \
                       // both-warning {{ISO C++17 does not permit structured binding declaration in a condition}}
    ;
}
static_assert(g("x") == 'x');

namespace IfScope {
  struct Inc {
    int &a;
    constexpr Inc(int &a) : a(a) {}
    constexpr ~Inc() { ++a; }
  };

  constexpr int foo() {
    int a= 0;
    int b = 12;
    if (Inc{a}; true) {
      b += a;
    }
    return b;
  }
  static_assert(foo() == 13, "");
}

namespace IfScope2 {
  struct __bit_iterator {
    unsigned __ctz_;
  };
  constexpr void __fill_n_bool(__bit_iterator) {}

  constexpr void fill_n(__bit_iterator __first) {
    if (false)
      __fill_n_bool(__first);
    else
      __fill_n_bool(__first);
  }

  struct bitset{
    constexpr void reset() {
      auto m = __bit_iterator(8);
      fill_n(m);
    }
  };
  consteval bool foo() {
    bitset v;
    v.reset();
    return true;
  }
  static_assert(foo());
}
