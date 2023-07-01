// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify -std=c++20 %s
// RUN: %clang_cc1 -verify=ref -std=c++20 %s

constexpr int a = 12;
constexpr int f = [c = a]() { return c; }();
static_assert(f == a);


constexpr int inc() {
  int a = 10;
  auto f = [&a]() {
    ++a;
  };

  f();f();

  return a;
}
static_assert(inc() == 12);

constexpr int add(int a, int b) {
  auto doIt = [a, b](int c) {
    return a + b + c;
  };

  return doIt(2);
}
static_assert(add(4, 5) == 11);


constexpr int add2(int a, int b) {
  auto doIt = [a, b](int c) {
    auto bar = [a]() { return a; };
    auto bar2 = [b]() { return b; };

    return bar() + bar2() + c;
  };

  return doIt(2);
}
static_assert(add2(4, 5) == 11);


constexpr int div(int a, int b) {
  auto f = [=]() {
    return a / b; // expected-note {{division by zero}} \
                  // ref-note {{division by zero}}
  };

  return f(); // expected-note {{in call to '&f->operator()()'}} \
              // ref-note {{in call to 'f.operator()()'}}
}
static_assert(div(8, 2) == 4);
static_assert(div(8, 0) == 4); // expected-error {{not an integral constant expression}} \
                               // expected-note {{in call to 'div(8, 0)'}} \
                               // ref-error {{not an integral constant expression}} \
                               // ref-note {{in call to 'div(8, 0)'}}


struct F {
  float f;
};

constexpr float captureStruct() {
  F someF = {1.0};

  auto p = [someF]() {
    return someF.f;
  };

  return p();
}

static_assert(captureStruct() == 1.0);


int constexpr FunCase() {
    return [x = 10] {
       decltype(x) y; // type int b/c not odr use
                      // refers to original init-capture
       auto &z = x; // type const int & b/c odr use
                     // refers to lambdas copy of x
        y = 10; // Ok
        //z = 10; // Ill-formed
        return y;
    }();
}

constexpr int WC = FunCase();


namespace LambdaParams {
  template<typename T>
  constexpr void callThis(T t) {
    return t();
  }

  constexpr int foo() {
    int a = 0;
    auto f = [&a]() { ++a; };

    callThis(f);

    return a;
  }
  /// FIXME: This should work in the new interpreter.
  static_assert(foo() == 1); // expected-error {{not an integral constant expression}}
}

