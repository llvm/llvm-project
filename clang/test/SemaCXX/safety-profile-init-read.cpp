// All violations share one TU with a leading unrelated error: the early error
// disables the analysis-based-warnings pass for later functions, so this also
// verifies that an enforced CFG-uninit profile keeps diagnosing afterwards.
// The DEMOTE run additionally enforces test::uninit_read to exercise profile
// table ordering, which would otherwise change the std::init-only diagnostics.
// RUN: %clang_cc1 -fsyntax-only -verify=expected,common -fprofiles -std=c++23 -Wno-uninitialized %s
// RUN: %clang_cc1 -fsyntax-only -verify=demote,common -fprofiles -fprofiles-test-profiles -std=c++23 -Wno-uninitialized -DDEMOTE %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles,common -std=c++23 -Wno-uninitialized %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];
// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::other)]];
#ifdef DEMOTE
[[profiles::enforce(test::uninit_read)]];
#endif

namespace std { enum class byte : unsigned char {}; }

int leading_unrelated_error = undeclared_identifier;
// common-error@-1 {{use of undeclared identifier 'undeclared_identifier'}}

// The always-compiled suppress tests suppress both std::init and
// test::uninit_read so the function-under-test demonstrates std::init behavior
// in isolation under either run.
// no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init)]] [[profiles::suppress(test::uninit_read)]]
void test_suppress_decl() {
  int x [[uninit]];
  int y = x;
  (void)y;
}

void test_suppress_stmt_inner() {
  int x [[uninit]];
  // no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] [[profiles::suppress(test::uninit_read)]] {
    int y = x;
    (void)y;
  }
}

void test_suppress_var_init() {
  int x [[uninit]];
  // no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] [[profiles::suppress(test::uninit_read)]]
  int y = x;
  (void)y;
}

void test_suppress_rule_targeted() {
  int x [[uninit]];
  // no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
  // no-profiles-warning@+2 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]]
  [[profiles::suppress(test::uninit_read)]] {
    int y = x;
    (void)y;
  }
}

void test_marker_then_write_then_read() {
  int x [[uninit]];
  x = 7;
  int y = x;
  (void)y;
}

void test_param(int p) {
  int y = p;
  (void)y;
}

#ifndef DEMOTE
void test_marker_does_not_excuse_read() {
  int x [[uninit]]; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
  (void)y;
}

void test_suppress_stmt_outer() {
  int x [[uninit]]; // expected-note {{variable 'x' is declared here}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] {
    int y = x;
    (void)y;
  }
  int z = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
  (void)z;
}

template <typename T>
T template_uninit() {
  T x [[uninit]]; // expected-note {{variable 'x' is declared here}}
  return x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
}
void instantiate_template_uninit() {
  template_uninit<int>(); // expected-note {{in instantiation of function template specialization 'template_uninit<int>' requested here}}
}

void test_selective_suppress() {
  int x [[uninit]]; // expected-note {{variable 'x' is declared here}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::other)]] {
    int y = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
    (void)y;
  }
}

void test_decl_suppress_does_not_extend() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int x [[uninit]]; // expected-note {{variable 'x' is declared here}}
  int y = x; // expected-error {{variable 'x' is read before initialization under profile 'std::init'}}
  (void)y;
}

// std::byte may be read while uninitialized (paper section 4), so std::init
// does not diagnose a read of an uninitialized std::byte.
void test_byte_read_exempt() {
  std::byte b [[uninit]];
  std::byte c = b;
  (void)c;
}
#endif

#ifdef DEMOTE
// With both test::uninit_read and std::init enforced, table order makes
// test::uninit_read fire first; suppressing it at the use site lets the
// std::init diagnostic surface.
void test_demote_test_profile() {
  int x [[uninit]]; // demote-note {{variable 'x' is declared here}}
  [[profiles::suppress(test::uninit_read)]] {
    int y = x; // demote-error {{variable 'x' is read before initialization under profile 'std::init'}}
    (void)y;
  }
}

// The std::byte exemption is std::init-only: test::uninit_read still diagnoses
// a read of an uninitialized std::byte.
void test_byte_not_exempt_under_test_profile() {
  std::byte b [[uninit]]; // demote-note {{variable 'b' is declared here}}
  std::byte c = b; // demote-error {{variable 'b' is read before initialization under profile 'test::uninit_read'}}
  (void)c;
}
#endif
