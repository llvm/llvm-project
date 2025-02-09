// RUN: %clang_cc1 -fsyntax-only -verify -Wunused %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 -Wunused %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wunused %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wunused %s

// PR4103 : Make sure we don't get a bogus unused expression warning
namespace PR4103 {
  class APInt {
    char foo; // expected-warning {{private field 'foo' is not used}}
  };
  class APSInt : public APInt {
    char bar; // expected-warning {{private field 'bar' is not used}}
  public:
    APSInt &operator=(const APSInt &RHS);
  };

  APSInt& APSInt::operator=(const APSInt &RHS) {
    APInt::operator=(RHS);
    return *this;
  }

  template<typename T>
  struct X {
    X();
  };

  void test() {
    X<int>();
  }
}

namespace derefvolatile {
  void f(volatile char* x) {
    *x;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{expression result unused; assign into a variable to force a volatile load}}
#endif
    (void)*x;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{expression result unused; assign into a variable to force a volatile load}}
#endif
    volatile char y = 10;
    (void)y; // don't warn here, because it's a common pattern.
  }
}

namespace AnonObject {
  struct Foo {
    Foo(const char* const message);
    ~Foo();
  };
  void f() {
    Foo("Hello World!");  // don't warn
    int(1); // expected-warning {{expression result unused}}
  }
}

// Test that constructing an object (which may have side effects) with
// constructor arguments which are dependent doesn't produce an unused value
// warning.
namespace UnresolvedLookup {
  struct Foo {
    Foo(int i, int j);
  };
  template <typename T>
  struct Bar {
    void f(T t) {
      Foo(t, 0);  // no warning
    }
  };
}

#if __cplusplus >= 201703L
namespace PR33839 {
  void a() {
    struct X { int a, b; } x;
    auto [a, b] = x; // expected-warning {{unused variable '[a, b]'}}
    auto [c, d] = x;
    (void)d;
  }

  template<typename T> void f() {
    struct A { int n; } a[1];
    for (auto [x] : a) {
      (void)x;
    }
    auto [y] = a[0]; // expected-warning {{unused}}
  }
  template<bool b> void g() {
    struct A { int n; } a[1];
    for (auto [x] : a) {
      if constexpr (b)
        (void)x;
    }

    auto [y] = a[0];
    if constexpr (b)
      (void)y; // ok, even when b == false
  }
  template<typename T> void h() {
    struct A { int n; } a[1];
    for (auto [x] : a) { // expected-warning {{unused variable '[x]'}}
    }
  }
  void use() {
    f<int>(); // expected-note {{instantiation of}}
    g<true>();
    g<false>();
    h<int>(); // expected-note {{instantiation of}}
  }
}

namespace maybe_unused_binding {

void test() {
  struct X { int a, b; } x;
  auto [a [[maybe_unused]], b] = x; // expected-warning {{an attribute specifier sequence attached to a structured binding declaration is a C++2c extension}}
}

}

#endif
