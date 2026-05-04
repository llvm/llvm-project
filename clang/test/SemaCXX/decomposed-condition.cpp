// RUN: %clang_cc1 -std=c++17 -Wno-c++26-extensions -verify %s
// RUN: %clang_cc1 -std=c++17 -Wno-c++26-extensions -verify %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++2c -Wpre-c++26-compat -verify=cxx26,expected %s
// RUN: %clang_cc1 -std=c++2c -Wpre-c++26-compat -verify=cxx26,expected %s -fexperimental-new-constant-interpreter

struct X {
  bool flag;
  int data;
  constexpr explicit operator bool() const {
    return flag;
  }
  constexpr operator int() const {
    return data;
  }
};

namespace CondInIf {
constexpr int f(X x) {
  if (auto [ok, d] = x) // cxx26-warning {{structured binding declaration in a condition is incompatible with C++ standards before C++2c}}
    return d + int(ok);
  else
    return d * int(ok);
  ok = {}; // expected-error {{use of undeclared identifier 'ok'}}
  d = {};  // expected-error {{use of undeclared identifier 'd'}}
}

static_assert(f({true, 2}) == 3);
static_assert(f({false, 2}) == 0);

constexpr char g(char const (&x)[2]) {
  if (auto &[a, b] = x) // cxx26-warning {{structured binding declaration in a condition is incompatible with C++ standards before C++2c}}
    return a;
  else
    return b;

  if (auto [a, b] = x) // expected-error {{an array type is not allowed here}} \
                       // cxx26-warning {{structured binding declaration in a condition is incompatible with C++ standards before C++2c}}
    ;
}

static_assert(g("x") == 'x');
} // namespace CondInIf

namespace CondInSwitch {
constexpr int f(int n) {
  switch (X s = {true, n}; auto [ok, d] = s) {
    // cxx26-warning@-1 {{structured binding declaration in a condition is incompatible with C++ standards before C++2c}}
    s = {};
  case 0:
    return int(ok);
  case 1:
    return d * 10;
  case 2:
    return d * 40;
  default:
    return 0;
  }
  ok = {}; // expected-error {{use of undeclared identifier 'ok'}}
  d = {};  // expected-error {{use of undeclared identifier 'd'}}
  s = {};  // expected-error {{use of undeclared identifier 's'}}
}

static_assert(f(0) == 1);
static_assert(f(1) == 10);
static_assert(f(2) == 80);
} // namespace CondInSwitch

namespace CondInWhile {
constexpr int f(int n) {
  int m = 1;
  while (auto [ok, d] = X{n > 1, n}) {
    // cxx26-warning@-1 {{structured binding declaration in a condition is incompatible with C++ standards before C++2c}}
    m *= d;
    --n;
  }
  return m;
  return ok; // expected-error {{use of undeclared identifier 'ok'}}
}

static_assert(f(0) == 1);
static_assert(f(1) == 1);
static_assert(f(4) == 24);
} // namespace CondInWhile

namespace CondInFor {
constexpr int f(int n) {
  int a = 1, b = 1;
  for (X x = {true, n}; auto &[ok, d] = x; --d) {
    // cxx26-warning@-1 {{structured binding declaration in a condition is incompatible with C++ standards before C++2c}}
    if (d < 2)
      ok = false;
    else {
      int x = b;
      b += a;
      a = x;
    }
  }
  return b;
  return d; // expected-error {{use of undeclared identifier 'd'}}
}

static_assert(f(0) == 1);
static_assert(f(1) == 1);
static_assert(f(2) == 2);
static_assert(f(5) == 8);
} // namespace CondInFor
