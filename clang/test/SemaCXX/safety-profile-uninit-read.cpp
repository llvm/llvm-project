// All violations share one TU with a leading unrelated error: the early error
// disables the analysis-based-warnings pass for later functions, so this also
// verifies that an enforced CFG-uninit profile keeps diagnosing afterwards.
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 -Wno-uninitialized %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::uninit_read)]];
// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::other)]];

int leading_unrelated_error = undeclared_identifier;
// expected-error@-1 {{use of undeclared identifier 'undeclared_identifier'}}
// no-profiles-error@-2 {{use of undeclared identifier 'undeclared_identifier'}}

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

void test_violation() {
  int x; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
  (void)y;
}

void test_suppress_stmt_outer() {
  int x; // expected-note {{variable 'x' is declared here}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::uninit_read)]] {
    int y = x;
    (void)y;
  }
  int z = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
  (void)z;
}

template <typename T>
T template_uninit() {
  T x; // expected-note {{variable 'x' is declared here}}
  return x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
}
void instantiate_template_uninit() {
  template_uninit<int>(); // expected-note {{in instantiation of function template specialization 'template_uninit<int>' requested here}}
}

void test_self_init_with_use() {
  int x = x; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
  (void)y;
}

void test_selective_suppress() {
  int x; // expected-note {{variable 'x' is declared here}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::other)]] {
    int y = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
    (void)y;
  }
}

// Suppress on a declaration is token-based: it covers the initializer of that
// declaration but not later uses that live in different declarations.
void test_decl_suppress_does_not_extend() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::uninit_read)]] int x; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'test::uninit_read'}}
  (void)y;
}
