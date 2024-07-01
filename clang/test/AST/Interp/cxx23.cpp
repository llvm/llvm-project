// UNSUPPORTED:  target={{.*}}-zos{{.*}}
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify=ref,ref20,all,all20 %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fcxx-exceptions -verify=ref,ref23,all,all23 %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify=expected20,all,all20 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fcxx-exceptions -verify=expected23,all,all23 %s -fexperimental-new-constant-interpreter

constexpr int f(int n) {  // all20-error {{constexpr function never produces a constant expression}}
  static const int m = n; // all-note {{control flows through the definition of a static variable}} \
                          // all20-note {{control flows through the definition of a static variable}} \
                          // all20-warning {{is a C++23 extension}}

  return m;
}
static_assert(f(0) == 0, ""); // all-error {{not an integral constant expression}} \
                              // all-note {{in call to}}

constexpr int g(int n) {        // all20-error {{constexpr function never produces a constant expression}}
  thread_local const int m = n; // all-note {{control flows through the definition of a thread_local variable}} \
                                // all20-note {{control flows through the definition of a thread_local variable}} \
                                // all20-warning {{is a C++23 extension}}
  return m;
}
static_assert(g(0) == 0, ""); // all-error {{not an integral constant expression}} \
                              // all-note {{in call to}}

constexpr int c_thread_local(int n) { // all20-error {{constexpr function never produces a constant expression}}
  static _Thread_local int m = 0;     // all20-note 2{{control flows through the definition of a thread_local variable}} \
                                      // all23-note {{control flows through the definition of a thread_local variable}} \
                                      // all20-warning {{is a C++23 extension}}
  return m;
}
static_assert(c_thread_local(0) == 0, ""); // all-error {{not an integral constant expression}} \
                                           // all-note {{in call to}}


constexpr int gnu_thread_local(int n) { // all20-error {{constexpr function never produces a constant expression}}
  static __thread int m = 0;            // all20-note 2{{control flows through the definition of a thread_local variable}} \
                                        // all23-note {{control flows through the definition of a thread_local variable}} \
                                        // all20-warning {{is a C++23 extension}}
  return m;
}
static_assert(gnu_thread_local(0) == 0, ""); // all-error {{not an integral constant expression}} \
                                             // all-note {{in call to}}

constexpr int h(int n) {  // all20-error {{constexpr function never produces a constant expression}}
  static const int m = n; // all20-note {{control flows through the definition of a static variable}} \
                          // all20-warning {{is a C++23 extension}}
  return &m - &m;
}

constexpr int i(int n) {        // all20-error {{constexpr function never produces a constant expression}}
  thread_local const int m = n; // all20-note {{control flows through the definition of a thread_local variable}} \
                                // all20-warning {{is a C++23 extension}}
  return &m - &m;
}

constexpr int j(int n) {
  if (!n)
    return 0;
  static const int m = n; // ref20-warning {{is a C++23 extension}} \
                          // expected20-warning {{is a C++23 extension}}
  return m;
}
constexpr int j0 = j(0);

constexpr int k(int n) {
  if (!n)
    return 0;
  thread_local const int m = n; // ref20-warning {{is a C++23 extension}} \
                                // expected20-warning {{is a C++23 extension}}

  return m;
}
constexpr int k0 = k(0);

namespace StaticLambdas {
  constexpr auto static_capture_constexpr() {
    char n = 'n';
    return [n] static { return n; }(); // expected23-error {{a static lambda cannot have any captures}} \
                                       // expected20-error {{a static lambda cannot have any captures}} \
                                       // expected20-warning {{are a C++23 extension}} \
                                       // expected20-warning {{is a C++23 extension}} \
                                       // ref23-error {{a static lambda cannot have any captures}} \
                                       // ref20-error {{a static lambda cannot have any captures}} \
                                       // ref20-warning {{are a C++23 extension}} \
                                       // ref20-warning {{is a C++23 extension}}
  }
  static_assert(static_capture_constexpr()); // expected23-error {{static assertion expression is not an integral constant expression}} \
                                             // expected20-error {{static assertion expression is not an integral constant expression}} \
                                             // ref23-error {{static assertion expression is not an integral constant expression}} \
                                             // ref20-error {{static assertion expression is not an integral constant expression}}

  constexpr auto capture_constexpr() {
    char n = 'n';
    return [n] { return n; }();
  }
  static_assert(capture_constexpr());
}

namespace StaticOperators {
  auto lstatic = []() static { return 3; };  // ref20-warning {{C++23 extension}} \
                                             // expected20-warning {{C++23 extension}}
  static_assert(lstatic() == 3, "");
  constexpr int (*f2)(void) = lstatic;
  static_assert(f2() == 3);

  struct S1 {
    constexpr S1() { // all20-error {{never produces a constant expression}}
      throw; // all-note {{not valid in a constant expression}} \
             // all20-note {{not valid in a constant expression}}
    }
    static constexpr int operator()() { return 3; } // ref20-warning {{C++23 extension}} \
                                                    // expected20-warning {{C++23 extension}}
  };
  static_assert(S1{}() == 3, ""); // all-error {{not an integral constant expression}} \
                                  // all-note {{in call to}}



}

int test_in_lambdas() {
  auto c = [](int n) constexpr {
    if (n == 0)
      return 0;
    else
      goto test; // all-note {{subexpression not valid in a constant expression}} \
                 // all20-warning {{use of this statement in a constexpr function is a C++23 extension}}
  test:
    return 1;
  };
  c(0);
  constexpr auto A = c(1); // all-error {{must be initialized by a constant expression}} \
                           // all-note {{in call to}}
  return 0;
}

/// PackIndexExpr.
template <auto... p>
struct check_ice {
    enum e {
        x = p...[0] // all-warning {{is a C++2c extension}}
    };
};
static_assert(check_ice<42>::x == 42);


namespace VirtualBases {
  namespace One {
    struct U { int n; };
    struct V : U { int n; };
    struct A : virtual V { int n; };
    struct Aa { int n; };
    struct B : virtual A, Aa {};
    struct C : virtual A, Aa {};
    struct D : B, C {};

    /// Calls the constructor of D.
    D d;
  }
}

namespace LabelGoto {
  constexpr int foo() { // all20-error {{never produces a constant expression}}
    a: // all20-warning {{use of this statement in a constexpr function is a C++23 extension}}
    goto a; // all20-note 2{{subexpression not valid in a constant expression}} \
            // ref23-note {{subexpression not valid in a constant expression}} \
            // expected23-note {{subexpression not valid in a constant expression}}

    return 1;
  }
  static_assert(foo() == 1, ""); // all-error {{not an integral constant expression}} \
                                 // all-note {{in call to}}
}

namespace ExplicitLambdaThis {
  constexpr auto f = [x = 3]<typename Self>(this Self self) { // all20-error {{explicit object parameters are incompatible with C++ standards before C++2b}}
      return x;
  };
  static_assert(f());
}

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less = {-1};
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};
}

namespace UndefinedThreeWay {
  struct A {
    friend constexpr std::strong_ordering operator<=>(const A&, const A&) = default; // all-note {{declared here}}
  };

  constexpr std::strong_ordering operator<=>(const A&, const A&) noexcept;
  constexpr std::strong_ordering (*test_a_threeway)(const A&, const A&) = &operator<=>;
  static_assert(!(*test_a_threeway)(A(), A())); // all-error {{static assertion expression is not an integral constant expression}} \
                                                // all-note {{undefined function 'operator<=>' cannot be used in a constant expression}}
}

/// FIXME: The new interpreter is missing the "initializer of q is not a constant expression" diagnostics.a
/// That's because the cast from void* to int* is considered fine, but diagnosed. So we don't consider
/// q to be uninitialized.
namespace VoidCast {
  constexpr void* p = nullptr;
  constexpr int* q = static_cast<int*>(p); // all-error {{must be initialized by a constant expression}} \
                                           // all-note {{cast from 'void *' is not allowed in a constant expression}} \
                                           // ref-note {{declared here}}
  static_assert(q == nullptr); // ref-error {{not an integral constant expression}} \
                               // ref-note {{initializer of 'q' is not a constant expression}}
}

namespace ExplicitLambdaInstancePointer {
  struct C {
      constexpr C(auto) { }
  };
  void foo() {
      constexpr auto b = [](this C) { return 1; }; // all20-error {{explicit object parameters are incompatible with C++ standards before C++2b}}
      constexpr int (*fp)(C) = b;
      static_assert(fp(1) == 1, "");
  }
}
