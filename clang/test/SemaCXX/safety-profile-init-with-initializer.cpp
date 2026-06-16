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

struct WithCtor { WithCtor(); };

void test_class_synthesized_init() {
  // The implicit default-constructor call is the initializer, so the marker
  // contradicts it even though there is no explicit initializer.
  WithCtor x [[uninitialized]]; // expected-error {{variable 'x' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}
  [[uninitialized]] WithCtor y; // expected-error {{variable 'y' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}
  (void)x; (void)y;
}

struct NoDefaultCtor { NoDefaultCtor() = delete; }; // expected-note {{'NoDefaultCtor' has been explicitly marked deleted here}} \
                                                    // no-profiles-note {{'NoDefaultCtor' has been explicitly marked deleted here}}

void test_failed_init_no_double_diag() {
  // Default-init fails and installs a RecoveryExpr placeholder. That is not a
  // user-written initializer, so R4 must not pile a spurious diagnostic on top
  // of the real error (the absence of an extra '...have an initializer...'
  // diagnostic here is the assertion).
  NoDefaultCtor z [[uninitialized]]; // expected-error {{call to deleted constructor of 'NoDefaultCtor'}} \
                                     // no-profiles-error {{call to deleted constructor of 'NoDefaultCtor'}}
  (void)z;
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

// [[profiles::suppress]] on a data member covers the uninit_with_initializer
// check that runs when its NSDMI is finalized, not just its parsing.
struct WithSuppressedNSDMI {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_with_initializer")]] int m [[uninitialized]] = 0; // OK: rule-targeted suppress
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int n [[uninitialized]] = 1;                                  // OK: whole-profile suppress
};

// A suppress on the enclosing record covers its members' NSDMIs.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] WithClassLevelSuppressedNSDMI {
  int m [[uninitialized]] = 0; // OK: suppressed by the class-level attribute
};

template <typename T>
void template_marker_with_init() {
  T x [[uninitialized]] = T{}; // expected-error 2 {{variable 'x' cannot be both '[[uninitialized]]' and have an initializer under profile 'std::init'}}
  (void)x;
}
template void template_marker_with_init<int>(); // expected-note {{in instantiation of function template specialization 'template_marker_with_init<int>' requested here}}
