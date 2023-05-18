// RUN: %clang_cc1 -verify -std=c++1y %s

// Example from the standard.
namespace X {
  void p() {
    q(); // expected-error {{undeclared}}
    extern void q();
  }
  void middle() {
    q(); // expected-error {{undeclared}}
  }
  void q() { /*...*/ }
  void bottom() {
    q();
  }
}
int q();

namespace Test1 {
  void f() {
    extern int a; // expected-note {{previous}}
    int g(void); // expected-note {{previous}}
  }
  double a; // expected-error {{different type: 'double' vs 'int'}}
  double g(); // expected-error {{differ only in their return type}}
}

namespace Test2 {
  void f() {
    extern int a; // expected-note {{previous}}
    int g(void); // expected-note {{previous}}
  }
  void h() {
    extern double a; // expected-error {{different type: 'double' vs 'int'}}
    double g(void); // expected-error {{differ only in their return type}}
  }
}

namespace Test3 {
  constexpr void (*f())() {
    void h();
    return &h;
  }
  constexpr void (*g())() {
    void h();
    return &h;
  }
  static_assert(f() == g(), "");
}

namespace Test4 {
  template<typename T>
  constexpr void (*f())() {
    void h();
    return &h;
  }
  static_assert(f<int>() == f<char>(), "");
  void h();
  static_assert(f<int>() == &h, "");
}

namespace Test5 {
  constexpr auto f() -> void (*)() {
    void g();
    struct X {
      friend void g();
      static constexpr auto h() -> void (*)() { return g; }
    };
    return X::h();
  }
  void g();
  static_assert(f() == g, "");
}
