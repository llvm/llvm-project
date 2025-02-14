// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s

/// FIXME: Slight difference in diagnostic output here.

struct Foo {
  int a;
};

constexpr int dead1() {

  Foo *F2 = nullptr;
  {
    Foo F{12}; // expected-note {{declared here}}
    F2 = &F;
  } // Ends lifetime of F.

  return F2->a; // expected-note {{read of variable whose lifetime has ended}} \
                // ref-note {{read of object outside its lifetime is not allowed in a constant expression}}
}
static_assert(dead1() == 1, ""); // both-error {{not an integral constant expression}} \
                                 // both-note {{in call to}}


struct S {
  int &&r; // both-note {{reference member declared here}}
  int t;
  constexpr S() : r(0), t(r) {} // both-error {{reference member 'r' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}} \
                                // ref-note {{read of object outside its lifetime is not allowed in a constant expression}} \
                                // expected-note {{temporary created here}} \
                                // expected-note {{read of temporary whose lifetime has ended}}
};
constexpr int k1 = S().t; // both-error {{must be initialized by a constant expression}} \
                          // ref-note {{in call to}} \
                          // expected-note {{in call to}}


namespace MoveFnWorks {
  template<typename T> constexpr T &&ref(T &&t) { return (T&&)t; }

  struct Buf {};

  struct A {
    constexpr A(Buf &buf) : buf(buf) { }
    Buf &buf;
  };

  constexpr bool dtor_calls_dtor() {
    struct B {
      A &&d;
      constexpr B(Buf &buf) : d(ref(A(buf))) {}
    };

    Buf buf;
    {
      B b(buf);
    }

    return true;
  }
  static_assert(dtor_calls_dtor(), "");
}

namespace PrimitiveMoveFn {
  /// This used to crash.
  void test() {
    const float y = 100;
    const float &x = y;
  }
}
