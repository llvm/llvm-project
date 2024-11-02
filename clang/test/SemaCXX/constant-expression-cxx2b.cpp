// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,cxx2a %s -fcxx-exceptions -triple=x86_64-linux-gnu -Wno-c++2b-extensions
// RUN: %clang_cc1 -std=c++2b -fsyntax-only -verify=expected,cxx2b %s -fcxx-exceptions -triple=x86_64-linux-gnu -Wpre-c++2b-compat

struct NonLiteral { // cxx2a-note {{'NonLiteral' is not literal}} \
                    // cxx2b-note 2{{'NonLiteral' is not literal}}
  NonLiteral() {}
};

struct Constexpr{};

#if __cplusplus > 202002L

constexpr int f(int n) {  // expected-error {{constexpr function never produces a constant expression}}
  static const int m = n; // expected-note {{control flows through the definition of a static variable}} \
                          // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return m;
}
constexpr int g(int n) {        // expected-error {{constexpr function never produces a constant expression}}
  thread_local const int m = n; // expected-note {{control flows through the definition of a thread_local variable}} \
                                // cxx2b-warning {{definition of a thread_local variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return m;
}

constexpr int c_thread_local(int n) { // expected-error {{constexpr function never produces a constant expression}}
  static _Thread_local int m = 0;     // expected-note {{control flows through the definition of a thread_local variable}} \
                                      // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return m;
}

constexpr int gnu_thread_local(int n) { // expected-error {{constexpr function never produces a constant expression}}
  static __thread int m = 0;            // expected-note {{control flows through the definition of a thread_local variable}} \
                                        // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return m;
}

constexpr int h(int n) {  // expected-error {{constexpr function never produces a constant expression}}
  static const int m = n; // expected-note {{control flows through the definition of a static variable}} \
                          // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return &m - &m;
}
constexpr int i(int n) {        // expected-error {{constexpr function never produces a constant expression}}
  thread_local const int m = n; // expected-note {{control flows through the definition of a thread_local variable}} \
                                 // cxx2b-warning {{definition of a thread_local variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return &m - &m;
}

constexpr int j(int n) {
  if (!n)
    return 0;
  static const int m = n; // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return m;
}
constexpr int j0 = j(0);

constexpr int k(int n) {
  if (!n)
    return 0;
  thread_local const int m = n; // cxx2b-warning {{definition of a thread_local variable in a constexpr function is incompatible with C++ standards before C++2b}}

  return m;
}
constexpr int k0 = k(0);

constexpr int j_evaluated(int n) {
  if (!n)
    return 0;
  static const int m = n; // expected-note {{control flows through the definition of a static variable}} \
                          // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return m;
}

constexpr int je = j_evaluated(1); // expected-error {{constexpr variable 'je' must be initialized by a constant expression}}  \
                                   // expected-note {{in call}}

constexpr int k_evaluated(int n) {
  if (!n)
    return 0;
  thread_local const int m = n; // expected-note {{control flows through the definition of a thread_local variable}} \
                                // cxx2b-warning {{definition of a thread_local variable in a constexpr function is incompatible with C++ standards before C++2b}}

  return m;
}

constexpr int ke = k_evaluated(1); // expected-error {{constexpr variable 'ke' must be initialized by a constant expression}} \
                                   // expected-note {{in call}}

constexpr int static_constexpr() {
  static constexpr int m = 42;     // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
  static constexpr Constexpr foo; // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return m;
}

constexpr int thread_local_constexpr() {
  thread_local constexpr int m = 42; // cxx2b-warning {{definition of a thread_local variable in a constexpr function is incompatible with C++ standards before C++2b}}
  thread_local constexpr Constexpr foo; // cxx2b-warning {{definition of a thread_local variable in a constexpr function is incompatible with C++ standards before C++2b}}
  return m;
}

constexpr int non_literal(bool b) {
  if (!b)
    return 0;
  NonLiteral n; // cxx2b-warning {{definition of a variable of non-literal type in a constexpr function is incompatible with C++ standards before C++2b}}
}

constexpr int non_literal_1 = non_literal(false);

namespace eval_goto {

constexpr int f(int x) {
  if (x) {
    return 0;
  } else {
    goto test; // expected-note {{subexpression not valid in a constant expression}} \
               // cxx2b-warning {{use of this statement in a constexpr function is incompatible with C++ standards before C++2b}}
  }
test:
  return 0;
}

int a = f(0);
constexpr int b = f(0); // expected-error {{must be initialized by a constant expression}} \
                        // expected-note {{in call to 'f(0)'}}
constexpr int c = f(1);

constexpr int label() {

test: // cxx2b-warning {{use of this statement in a constexpr function is incompatible with C++ standards before C++2b}}
  return 0;
}

constexpr int d = label();

} // namespace eval_goto

#endif

// Test that explicitly constexpr lambdas behave correctly,
// This is to be contrasted with the test for implicitly constexpr lambdas below.
int test_in_lambdas() {
  auto a = []() constexpr {
    static const int m = 32; // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
    return m;
  };

  auto b = [](int n) constexpr {
    if (!n)
      return 0;
    static const int m = n; // cxx2b-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++2b}}
    return m;
  }
  (1);

  auto c = [](int n) constexpr {
    if (!n)
      return 0;
    else
      goto test; // expected-note {{subexpression not valid in a constant expression}} \
                 // cxx2b-warning {{use of this statement in a constexpr function is incompatible with C++ standards before C++2b}}
  test:
    return 1;
  };
  c(0);
  constexpr auto c_error = c(1); // expected-error {{constexpr variable 'c_error' must be initialized by a constant expression}} \
                                 // expected-note {{in call to}}

  auto non_literal = [](bool b) constexpr {
    if (!b)
      NonLiteral n; // cxx2b-note {{non-literal type 'NonLiteral' cannot be used in a constant expression}} \
                    // cxx2a-error {{variable of non-literal type 'NonLiteral' cannot be defined in a constexpr function before C++2b}} \
                    // cxx2b-warning {{definition of a variable of non-literal type in a constexpr function is incompatible with C++ standards before C++2b}}
    return 0;
  };

#if __cplusplus > 202002L
  constexpr auto non_literal_ko = non_literal(false); // cxx2b-error {{constexpr variable 'non_literal_ko' must be initialized by a constant expression}} \
                                                      // cxx2b-note {{in call}}

  constexpr auto non_literal_ok = non_literal(true);
#endif
}

// Test whether lambdas are correctly treated as implicitly constexpr under the
// relaxed C++23 rules (and similarly as not implicitly constexpr under the
// C++20 rules).
int test_lambdas_implicitly_constexpr() {

  auto b = [](int n) { // cxx2a-note 2{{declared here}}
    if (!n)
      return 0;
    static const int m = n; // cxx2b-note {{control flows through the definition of a static variable}}
    return m;
  };

  auto b1 = b(1);
  constexpr auto b2 = b(0); // cxx2a-error {{must be initialized by a constant expression}} \
                            // cxx2a-note {{non-constexpr function}}

  constexpr auto b3 = b(1); // expected-error{{constexpr variable 'b3' must be initialized by a constant expression}} \
                            // cxx2a-note {{non-constexpr function}} \
                            // cxx2b-note {{in call}}

  auto c = [](int n) { // cxx2a-note 2{{declared here}}
    if (!n)
      return 0;
    else
      goto test; // cxx2b-note {{subexpression not valid in a constant expression}}
  test:
    return 1;
  };
  c(0);
  constexpr auto c_ok = c(0); // cxx2a-error {{must be initialized by a constant expression}} \
                              // cxx2a-note {{non-constexpr function}}

  constexpr auto c_error = c(1); // expected-error {{constexpr variable 'c_error' must be initialized by a constant expression}} \
                                 // cxx2a-note {{non-constexpr function}} \
                                 // cxx2b-note {{in call to}}

  auto non_literal = [](bool b) { // cxx2a-note 2{{declared here}}
    if (b)
      NonLiteral n; // cxx2b-note {{non-literal type 'NonLiteral' cannot be used in a constant expression}}
    return 0;
  };

  constexpr auto non_literal_ko = non_literal(true); // expected-error {{constexpr variable 'non_literal_ko' must be initialized by a constant expression}} \
                                                     // cxx2a-note {{non-constexpr function}} \
                                                     // cxx2b-note {{in call}}

  constexpr auto non_literal_ok = non_literal(false); // cxx2a-error {{must be initialized by a constant expression}} \
                                                      // cxx2a-note {{non-constexpr function}}
}

template <typename T>
constexpr auto dependent_var_def_lambda() {
  return [](bool b) { // cxx2a-note {{declared here}}
    if (!b)
      T t;
    return 0;
  };
}

constexpr auto non_literal_valid_in_cxx2b = dependent_var_def_lambda<NonLiteral>()(true); // \
    // cxx2a-error {{constexpr variable 'non_literal_valid_in_cxx2b' must be initialized by a constant expression}} \
    // cxx2a-note {{non-constexpr function}}


constexpr double evaluate_static_constexpr() {
  struct Constexpr{
    constexpr double f() const {
      return 42;
    }
  };
  thread_local constexpr Constexpr t; // cxx2b-warning {{before C++2b}}
  static constexpr Constexpr s; // cxx2b-warning {{before C++2b}}
  return t.f() + s.f();
}
static_assert(evaluate_static_constexpr() == 84);
