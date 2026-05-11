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
[[profiles::enforce(std::init)]];
// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::other)]];
#if CASE == 4
[[profiles::enforce(test::uninit_read)]];
#endif

// Cases that never diagnose are always compiled.

// CASE=4 also enforces test::uninit_read; the always-compiled suppress tests
// suppress both so the function-under-test demonstrates std::init behavior in
// isolation.
// no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init)]] [[profiles::suppress(test::uninit_read)]]
void test_suppress_decl() {
  int x [[uninitialized]];
  int y = x;
  (void)y;
}

void test_suppress_stmt_inner() {
  int x [[uninitialized]];
  // no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] [[profiles::suppress(test::uninit_read)]] {
    int y = x;
    (void)y;
  }
}

void test_suppress_var_init() {
  int x [[uninitialized]];
  // no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] [[profiles::suppress(test::uninit_read)]]
  int y = x;
  (void)y;
}

void test_suppress_rule_targeted() {
  int x [[uninitialized]];
  // no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
  // no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]]
  [[profiles::suppress(test::uninit_read)]] {
    int y = x;
    (void)y;
  }
}

void test_marker_then_write_then_read() {
  int x [[uninitialized]];
  x = 7;
  int y = x;
  (void)y;
}

void test_param(int p) {
  int y = p;
  (void)y;
}

#if CASE == 1
void test_marker_does_not_excuse_read() {
  int x [[uninitialized]]; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
  (void)y;
}
#endif

#if CASE == 2
void test_suppress_stmt_outer() {
  int x [[uninitialized]]; // expected-note {{variable 'x' is declared here}}
  [[profiles::suppress(std::init)]] {
    int y = x;
    (void)y;
  }
  int z = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
  (void)z;
}
#endif

#if CASE == 3
template <typename T>
T template_uninit() {
  T x [[uninitialized]]; // expected-note {{variable 'x' is declared here}}
  return x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
}
void instantiate_template_uninit() {
  template_uninit<int>(); // expected-note {{in instantiation of function template specialization 'template_uninit<int>' requested here}}
}
#endif

#if CASE == 4
// When both test::uninit_read and std::init are enforced (the conditional
// enforce above adds test::uninit_read for this case), table order in
// CFGUninitProfiles makes test::uninit_read fire first. Suppressing it at
// the use site lets the std::init diagnostic surface.
void test_demote_test_profile() {
  int x [[uninitialized]]; // expected-note {{variable 'x' is declared here}}
  [[profiles::suppress(test::uninit_read)]] {
    int y = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
    (void)y;
  }
}
#endif

#if CASE == 5
void test_selective_suppress() {
  int x [[uninitialized]]; // expected-note {{variable 'x' is declared here}}
  [[profiles::suppress(test::other)]] {
    int y = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
    (void)y;
  }
}
#endif

#if CASE == 6
void test_decl_suppress_does_not_extend() {
  [[profiles::suppress(std::init)]] int x [[uninitialized]]; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
  (void)y;
}
#endif
