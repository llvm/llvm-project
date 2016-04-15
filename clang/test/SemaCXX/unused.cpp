// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// PR4103 : Make sure we don't get a bogus unused expression warning
namespace PR4103 {
  class APInt {
    char foo;
  };
  class APSInt : public APInt {
    char bar;
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

// <rdar://problem/12359208>
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
