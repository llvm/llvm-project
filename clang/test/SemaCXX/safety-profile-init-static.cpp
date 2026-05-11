// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

int runtime();   // expected-note {{declared here}}
                 // no-profiles-note@-1 {{declared here}}
constexpr int compile_time() { return 7; }

int g_const = 0;
int g_constexpr = compile_time();
int g_array_const[3] = {1, 2, 3};
int g_runtime = runtime();              // expected-error {{non-local variable 'g_runtime' requires constant initialization under profile 'std::init'}}
int g_runtime_array[3] = {1, runtime(), 3}; // expected-error {{non-local variable 'g_runtime_array' requires constant initialization under profile 'std::init'}}

constinit int g_ci = 0;
// The constinit hard error fires regardless of -fprofiles.
constinit int g_ci_runtime = runtime();
// expected-error@-1 {{variable does not have a constant initializer}}
// expected-note@-2 {{required by 'constinit' specifier here}}
// expected-note@-3 {{non-constexpr function 'runtime' cannot be used in a constant expression}}
// no-profiles-error@-4 {{variable does not have a constant initializer}}
// no-profiles-note@-5 {{required by 'constinit' specifier here}}
// no-profiles-note@-6 {{non-constexpr function 'runtime' cannot be used in a constant expression}}

namespace inside {
  int n_runtime = runtime();            // expected-error {{non-local variable 'n_runtime' requires constant initialization under profile 'std::init'}}
}

void test_locals() {
  int x = runtime();
  static int s = runtime();
  thread_local int t = runtime();
  (void)x; (void)s; (void)t;
}

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "static_runtime_init")]]
int g_suppressed = runtime();

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init)]]
int g_suppressed_all = runtime();

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::other)]]
int g_wrong_suppress = runtime();       // expected-error {{non-local variable 'g_wrong_suppress' requires constant initialization under profile 'std::init'}}
