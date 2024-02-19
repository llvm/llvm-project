// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both -std=c++20 %s
// RUN: %clang_cc1 -verify=ref,both -std=c++20 %s

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
    return a / b; // both-note {{division by zero}}
  };

  return f(); // expected-note {{in call to '&f->operator()()'}} \
              // ref-note {{in call to 'f.operator()()'}}
}
static_assert(div(8, 2) == 4);
static_assert(div(8, 0) == 4); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to 'div(8, 0)'}}


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
  static_assert(foo() == 1);
}

namespace StaticInvoker {
  constexpr int sv1(int i) {
    auto l = []() { return 12; };
    int (*fp)() = l;
    return fp();
  }
  static_assert(sv1(12) == 12);

  constexpr int sv2(int i) {
    auto l = [](int m, float f, void *A) { return m; };
    int (*fp)(int, float, void*) = l;
    return fp(i, 4.0f, nullptr);
  }
  static_assert(sv2(12) == 12);

  constexpr int sv3(int i) {
    auto l = [](int m, const int &n) { return m; };
    int (*fp)(int, const int &) = l;
    return fp(i, 3);
  }
  static_assert(sv3(12) == 12);

  constexpr int sv4(int i) {
    auto l = [](int &m) { return m; };
    int (*fp)(int&) = l;
    return fp(i);
  }
  static_assert(sv4(12) == 12);

  constexpr int sv5(int i) {
    struct F { int a; float f; };
    auto l = [](int m, F f) { return m; };
    int (*fp)(int, F) = l;
    return fp(i, F{12, 14.0});
  }
  static_assert(sv5(12) == 12);

  constexpr int sv6(int i) {
    struct F { int a;
      constexpr F(int a) : a(a) {}
    };

    auto l = [](int m) { return F(12); };
    F (*fp)(int) = l;
    F f = fp(i);

    return fp(i).a;
  }
  static_assert(sv6(12) == 12);


  /// A generic lambda.
  auto GL = [](auto a) { return a; };
  constexpr char (*fp2)(char) = GL;
  static_assert(fp2('3') == '3', "");

  struct GLS {
    int a;
  };
  auto GL2 = [](auto a) { return GLS{a}; };
  constexpr GLS (*fp3)(char) = GL2;
  static_assert(fp3('3').a == '3', "");
}

namespace LambdasAsParams {
  template<typename F>
  constexpr auto call(F f) {
    return f();
  }
  static_assert(call([](){ return 1;}) == 1);
  static_assert(call([](){ return 2;}) == 2);


  constexpr unsigned L = call([](){ return 12;});
  static_assert(L == 12);


  constexpr float heh() {
    auto a = []() {
      return 1.0;
    };

    return static_cast<float>(a());
  }
  static_assert(heh() == 1.0);
}

namespace ThisCapture {
  class Foo {
  public:
    int b = 32;
    int a;

    constexpr Foo() : a([this](){ return b + 1;}()) {}

    constexpr int Aplus2() const {
      auto F = [this]() {
        return a + 2;
      };

      return F();
    }
  };
  constexpr Foo F;
  static_assert(F.a == 33, "");
  static_assert(F.Aplus2() == (33 + 2), "");
}

namespace GH62611 {
  template <auto A = [](auto x){}>
  struct C {
    static constexpr auto B = A;
  };

  int test() {
    C<>::B(42);
    return 0;
  }
}

namespace LambdaToAPValue {
  void wrapper() {
    constexpr auto f = []() constexpr {
      return 0;
    };

    constexpr auto g = [f]() constexpr {
      return f();
    };
    static_assert(g() == f(), "");
  }
}
