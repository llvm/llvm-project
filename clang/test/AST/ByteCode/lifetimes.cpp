// RUN: %clang_cc1 -verify=expected,both -std=c++20 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify=ref,both      -std=c++20 %s

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
                          // both-note {{in call to}}


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

/// FIXME:
///  1) This doesn't work for parameters
///  2) We need to do this for all fields in composite scenarios
namespace PseudoDtor {
  typedef int I;
  constexpr bool foo() { // both-error {{never produces a constant expression}}
    {
      int a; // both-note {{destroying object 'a' whose lifetime has already ended}}
      a.~I();
    }
    return true;
  }

  int k;
  struct T {
    int n : (k.~I(), 1); // both-error {{constant expression}} \
                         // both-note {{visible outside that expression}}
  };
}

/// Diagnostic differences
namespace CallScope {
  struct Q {
    int n = 0;
    constexpr int f() const { return 0; }
  };
  constexpr Q *out_of_lifetime(Q q) { return &q; } // both-warning {{address of stack}} \
                                                   // expected-note 2{{declared here}}
  constexpr int k3 = out_of_lifetime({})->n; // both-error {{must be initialized by a constant expression}} \
                                             // expected-note {{read of variable whose lifetime has ended}} \
                                             // ref-note {{read of object outside its lifetime}}

  constexpr int k4 = out_of_lifetime({})->f(); // both-error {{must be initialized by a constant expression}} \
                                               // expected-note {{member call on variable whose lifetime has ended}} \
                                               // ref-note {{member call on object outside its lifetime}}
}

namespace ExprDoubleDestroy {
  template <typename T>
  constexpr bool test() {
    T{}.~T(); // both-note {{lifetime has already ended}}
    return true;
  }

  struct S { int x; };
  constexpr bool t = test<S>(); // both-error {{must be initialized by a constant expression}} \
                                // both-note {{in call to}}
}
