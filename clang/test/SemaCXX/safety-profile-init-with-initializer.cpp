// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

void test_postfix_eq() {
  int c [[uninitialized]] = 0; // expected-error {{variable 'c' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}
  (void)c;
}

void test_prefix_eq() {
  [[uninitialized]] int d = 7; // expected-error {{variable 'd' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}
  (void)d;
}

void test_brace_init() {
  int e [[uninitialized]] {}; // expected-error {{variable 'e' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}
  (void)e;
}

void test_paren_init() {
  int f [[uninitialized]] (3); // expected-error {{variable 'f' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}
  (void)f;
}

void test_marker_alone() {
  int x [[uninitialized]];
  (void)x;
}

void test_init_alone() {
  int x = 0;
  (void)x;
}

int g_marker_with_init [[uninitialized]] = 42; // expected-error {{variable 'g_marker_with_init' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}

void test_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_with_initializer")]]
  int x [[uninitialized]] = 0;
  (void)x;
}

void test_suppress_block() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] {
    int a [[uninitialized]] = 0;
    int b [[uninitialized]] = 1;
    (void)a; (void)b;
  }
}

template <typename T>
void template_marker_with_init() {
  T x [[uninitialized]] = T{}; // expected-error 2 {{variable 'x' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}
  (void)x;
}
template void template_marker_with_init<int>(); // expected-note {{in instantiation of function template specialization 'template_marker_with_init<int>' requested here}}
