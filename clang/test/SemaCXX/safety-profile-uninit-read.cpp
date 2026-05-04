// Each CASE selects one violation test so the analysis-based-warnings early
// exit on first error doesn't hide later cases.
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DCASE=0 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DCASE=1 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DCASE=2 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DCASE=3 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DCASE=4 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DCASE=5 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DCASE=6 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 -Wno-uninitialized -DCASE=0 %s

#if CASE == 0
// expected-no-diagnostics
#endif

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::uninit_read)]];
// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::other)]];

// Cases that never diagnose are always compiled.

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::uninit_read)]]
void test_suppress_decl() {
  int x;
  int y = x;
  (void)y;
}

void test_suppress_stmt_inner() {
  int x;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::uninit_read)]] {
    int y = x;
    (void)y;
  }
}

void test_suppress_var_init() {
  int x;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::uninit_read)]] int y = x;
  (void)y;
}

void test_discarded_branch() {
  int x;
  if constexpr (false) {
    int y = x;
    (void)y;
  }
}

void test_unevaluated() {
  int x;
  using T = decltype(x);
  (void)sizeof(x);
  (void)static_cast<T>(0);
}

void test_param(int p) {
  int y = p;
  (void)y;
}

int g_static;
void test_global() {
  int y = g_static;
  (void)y;
}

void test_self_init_no_use() {
  int x = x;
  (void)&x;
}

void test_suppress_self_init() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::uninit_read)]] int x = x;
  (void)&x;
}

#if CASE == 1
void test_violation() {
  int x; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
  (void)y;
}
#endif

#if CASE == 2
void test_suppress_stmt_outer() {
  int x; // expected-note {{variable 'x' is declared here}}
  [[profiles::suppress(test::uninit_read)]] {
    int y = x;
    (void)y;
  }
  int z = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
  (void)z;
}
#endif

#if CASE == 3
template <typename T>
T template_uninit() {
  T x; // expected-note {{variable 'x' is declared here}}
  return x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
}
void instantiate_template_uninit() {
  template_uninit<int>(); // expected-note {{in instantiation of function template specialization 'template_uninit<int>' requested here}}
}
#endif

#if CASE == 4
void test_self_init_with_use() {
  int x = x; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
  (void)y;
}
#endif

#if CASE == 5
void test_selective_suppress() {
  int x; // expected-note {{variable 'x' is declared here}}
  [[profiles::suppress(test::other)]] {
    int y = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
    (void)y;
  }
}
#endif

#if CASE == 6
// Suppress on a declaration is token-based: it covers the initializer of
// that declaration but not later uses that live in different declarations.
void test_decl_suppress_does_not_extend() {
  [[profiles::suppress(test::uninit_read)]] int x; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
  (void)y;
}
#endif
