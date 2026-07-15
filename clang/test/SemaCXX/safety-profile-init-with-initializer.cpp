// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

void test_postfix_eq() {
  int c [[uninit]] = 0; // expected-error {{variable 'c' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  (void)c;
}

void test_prefix_eq() {
  [[uninit]] int d = 7; // expected-error {{variable 'd' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  (void)d;
}

void test_brace_init() {
  int e [[uninit]] {}; // expected-error {{variable 'e' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  (void)e;
}

void test_paren_init() {
  int f [[uninit]] (3); // expected-error {{variable 'f' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  (void)f;
}

void test_marker_alone() {
  int x [[uninit]];
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
  WithCtor x [[uninit]]; // expected-error {{variable 'x' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  [[uninit]] WithCtor y; // expected-error {{variable 'y' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  (void)x; (void)y;
}

struct MarkedMemberAgg { int x [[uninit]]; };

void test_marked_member_agg() {
  // R4 stays factual: MarkedMemberAgg's default-initialization is a genuine
  // no-op that leaves x indeterminate, so marking the variable too is
  // consistent rather than a contradiction. (Honoring the member marker here
  // would wrongly fire uninit_with_initializer.)
  MarkedMemberAgg a [[uninit]];
  (void)a;
}

struct NoDefaultCtor { NoDefaultCtor() = delete; }; // expected-note {{'NoDefaultCtor' has been explicitly marked deleted here}} \
                                                    // no-profiles-note {{'NoDefaultCtor' has been explicitly marked deleted here}}

void test_failed_init_no_double_diag() {
  // Default-init fails and installs a RecoveryExpr placeholder. That is not a
  // user-written initializer, so R4 must not pile a spurious diagnostic on top
  // of the real error (the absence of an extra '...have an initializer...'
  // diagnostic here is the assertion).
  NoDefaultCtor z [[uninit]]; // expected-error {{call to deleted constructor of 'NoDefaultCtor'}} \
                                     // no-profiles-error {{call to deleted constructor of 'NoDefaultCtor'}}
  (void)z;
}

int g_marker_with_init [[uninit]] = 42; // expected-error {{variable 'g_marker_with_init' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}

void test_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_with_initializer")]]
  int x [[uninit]] = 0;
  (void)x;
}

void test_suppress_block() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] {
    int a [[uninit]] = 0;
    int b [[uninit]] = 1;
    (void)a; (void)b;
  }
}

// [[profiles::suppress]] on a data member covers the uninit_with_initializer
// check that runs when its NSDMI is finalized, not just its parsing.
struct WithSuppressedNSDMI {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_with_initializer")]] int m [[uninit]] = 0; // OK: rule-targeted suppress
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int n [[uninit]] = 1;                                  // OK: whole-profile suppress
};

// A suppress on the enclosing record covers its members' NSDMIs.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] WithClassLevelSuppressedNSDMI {
  int m [[uninit]] = 0; // OK: suppressed by the class-level attribute
};

// A profile rule fires on the instantiation, not on the template pattern, so a
// dependent [[uninit]]-with-initializer is diagnosed exactly once.
template <typename T>
void template_marker_with_init() {
  T x [[uninit]] = T{}; // expected-error {{variable 'x' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  (void)x;
}
template void template_marker_with_init<int>(); // expected-note {{in instantiation of function template specialization 'template_marker_with_init<int>' requested here}}

// A *non-dependent* declaration inside a template body is likewise diagnosed
// once -- at instantiation -- not on the pattern.
template <typename T>
void template_nondependent_with_init() {
  int x [[uninit]] = 0; // expected-error {{variable 'x' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  (void)x;
}
template void template_nondependent_with_init<int>(); // expected-note {{in instantiation of function template specialization 'template_nondependent_with_init<int>' requested here}}

// An uninstantiated template pattern is not yet a phase-7 entity, so no profile
// rule fires on it (no expected diagnostic here).
template <typename T>
void template_never_instantiated() {
  int x [[uninit]] = 0;
  (void)x;
}

// Default-initialization that is not a genuine no-op contradicts the marker
// exactly like a written initializer (paper §4.2 rule 2, §5.3): something is
// initialized, so the object is not left uninitialized.
struct MixedInner { MixedInner(); };
struct Mixed { int x; MixedInner s; };
struct WithNSDMIMember { int a; int b = 0; };
struct Polymorphic { virtual void f(); int x; };
struct TrivialAgg { int x; };

void test_default_init_not_noop() {
  Mixed s4 [[uninit]];            // expected-error {{variable 's4' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  WithNSDMIMember q [[uninit]];   // expected-error {{variable 'q' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  Polymorphic v [[uninit]];       // expected-error {{variable 'v' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  [[uninit]] Mixed arr[2];        // expected-error {{variable 'arr' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  TrivialAgg t [[uninit]];        // OK: a genuine no-op, 't.x' really is left
                                  // indeterminate
  (void)s4; (void)q; (void)v; (void)arr; (void)t;
}

// A `= P()` value-initialization zeroes the object -- an initialization the
// marker contradicts; the `= P{}` list form was already rejected (pinned
// here), and the two must not diverge.
struct P { int a; int b; };
void test_value_init() {
  P p [[uninit]] = P();   // expected-error {{variable 'p' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  P p2 [[uninit]] = P{};  // expected-error {{variable 'p2' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
  (void)p; (void)p2;
}

// An NSDMI in a class template is checked once, when its initializer is
// instantiated (here forced by initializing the specialization).
template <typename T>
struct WithTemplatedNSDMI {
  T m [[uninit]] = T{}; // expected-error {{member 'm' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
};
void use_templated_nsdmi() {
  WithTemplatedNSDMI<int> w = {}; // expected-note {{in instantiation of default member initializer 'WithTemplatedNSDMI<int>::m' requested here}}
  (void)w;
}
