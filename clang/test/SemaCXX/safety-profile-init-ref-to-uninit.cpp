// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// std::init / ref_to_uninit (paper §5): a [[ref_to_uninit]] pointer or
// reference must be bound to uninitialized memory, and an unmarked pointer or
// reference must not. This file exercises the rule at variable initialization.

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

int g_init = 0;
[[uninitialized]] int g_uninit;
[[uninitialized]] int g_uninit_arr[3];
[[ref_to_uninit]] int *allocate(int n);
[[ref_to_uninit]] void *alloc_void();

void test_pointer_target() {
  int *p1 [[ref_to_uninit]] = &g_uninit; // OK
  int *p2 [[ref_to_uninit]] = &g_init;   // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *p3 = &g_uninit;                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *p4 = &g_init;                      // OK
  int *p5 = nullptr;                      // OK
  (void)p1; (void)p2; (void)p3; (void)p4; (void)p5;
}

void test_pointer_sources() {
  int *base [[ref_to_uninit]] = &g_uninit;
  int *from_ptr [[ref_to_uninit]] = base;           // OK: base is [[ref_to_uninit]]
  int *from_array [[ref_to_uninit]] = g_uninit_arr; // OK: array-to-pointer decay
  int *from_call [[ref_to_uninit]] = allocate(3);   // OK: [[ref_to_uninit]] return
  int *bad_from_array = g_uninit_arr;               // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *bad_from_call = allocate(3);                 // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)base; (void)from_ptr; (void)from_array; (void)from_call;
  (void)bad_from_array; (void)bad_from_call;
}

// An explicit pointer-to-pointer cast of a [[ref_to_uninit]] pointer is itself
// [[ref_to_uninit]] (paper §4.3); the cast does not launder the marking.
void test_pointer_casts() {
  void *vp [[ref_to_uninit]] = &g_uninit;
  int *c1 [[ref_to_uninit]] = (int *)vp; // OK
  int *c2 = (int *)vp;                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  int *sc1 [[ref_to_uninit]] = static_cast<int *>(vp);      // OK
  int *sc2 = static_cast<int *>(vp);                         // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *rc1 [[ref_to_uninit]] = reinterpret_cast<int *>(vp); // OK
  int *rc2 = reinterpret_cast<int *>(vp);                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  // An initialized pointer round-tripped through a cast stays initialized.
  void *vi = &g_init;
  int *ci1 = (int *)vi;                   // OK
  int *ci2 [[ref_to_uninit]] = (int *)vi; // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}

  // Casting a [[ref_to_uninit]]-returning call result propagates the marking.
  int *cc1 [[ref_to_uninit]] = (int *)alloc_void(); // OK
  int *cc2 = (int *)alloc_void();                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  // Trust model: a pointer manufactured from an integer is not propagated.
  int *ti = reinterpret_cast<int *>(0xdeadbeef); // OK

  // Deref of a cast routes back through the pointer recognizer.
  int &dr [[ref_to_uninit]] = *(int *)vp; // OK

  (void)c1; (void)c2; (void)sc1; (void)sc2; (void)rc1; (void)rc2;
  (void)vi; (void)ci1; (void)ci2; (void)cc1; (void)cc2; (void)ti; (void)dr;
}

void test_reference_target() {
  int &r1 [[ref_to_uninit]] = g_uninit; // OK
  int &r2 [[ref_to_uninit]] = g_init;   // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int &r3 = g_uninit;                   // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int &r4 = g_init;                     // OK
  int *p [[ref_to_uninit]] = &g_uninit;
  int &r5 [[ref_to_uninit]] = *p;       // OK: *p denotes uninitialized storage
  // A [[ref_to_uninit]] reference is itself a source of uninitialized storage,
  // symmetric to the [[ref_to_uninit]] pointer-copy case.
  int &r6 [[ref_to_uninit]] = r1;       // OK: r1 refers to uninitialized memory
  int &r7 = r1;                         // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)r1; (void)r2; (void)r3; (void)r4; (void)r5; (void)r6; (void)r7; (void)p;
}

void test_assignment() {
  int *p [[ref_to_uninit]] = &g_uninit;
  p = &g_uninit; // OK
  p = &g_init;   // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *q = &g_init;
  q = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  q = &g_init;   // OK
  q = nullptr;   // OK
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { q = &g_uninit; } // OK: suppressed
  (void)p; (void)q;
}

void test_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ref_to_uninit")]] int *s = &g_uninit; // OK: suppressed
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int *s2 [[ref_to_uninit]] = &g_init;       // OK: suppressed
  (void)s; (void)s2;
}

void take_uninit_ptr(int *p [[ref_to_uninit]]);
void take_uninit_ref(int &r [[ref_to_uninit]]);
void take_ptr(int *p);
void take_ref(const int &r);
void uninitialized_fill(int *r [[ref_to_uninit]], int val);

void test_call_arguments() {
  take_uninit_ptr(&g_uninit);    // OK
  take_uninit_ptr(&g_init);      // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  take_uninit_ptr(g_uninit_arr); // OK: array-to-pointer decay
  take_uninit_ref(g_uninit);     // OK
  take_uninit_ref(g_init);       // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}

  take_ptr(&g_uninit);           // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  take_ptr(&g_init);             // OK
  take_ref(g_uninit);            // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  take_ref(g_init);              // OK

  // A [[ref_to_uninit]] reference argument matches a [[ref_to_uninit]] reference
  // parameter, and is rejected for an unmarked one.
  int &ru [[ref_to_uninit]] = g_uninit;
  take_uninit_ref(ru);           // OK
  take_ref(ru);                  // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)ru;

  // The worked example from paper §5.
  int a1[] = {1, 2, 3};
  [[uninitialized]] int a2[3];
  uninitialized_fill(a1, 10); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  uninitialized_fill(a2, 10); // OK

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { take_ptr(&g_uninit); } // OK: suppressed
}

struct Inner { int m; };

// Member access through a [[ref_to_uninit]] pointer denotes uninitialized
// storage. Arrow access (a->m, object *a) and explicit deref ((*a).m) must
// behave identically.
void test_member_through_pointer(Inner *ptr [[ref_to_uninit]]) {
  int *q1 = &ptr->m;                   // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *q2 = &(*ptr).m;                 // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *q3 [[ref_to_uninit]] = &ptr->m; // OK
  (void)q1; (void)q2; (void)q3;
}

struct WithFields {
  int *p1 [[ref_to_uninit]] = &g_uninit; // OK
  int *p2 [[ref_to_uninit]] = &g_init;   // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *p3 = &g_uninit;                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *p4 = &g_init;                      // OK
  int *p5 = nullptr;                      // OK
  int &r1 [[ref_to_uninit]] = g_uninit;  // OK
  int &r2 = g_uninit;                     // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};

// [[profiles::suppress]] on a data member must cover its initializer's
// finalization checks, not just the initializer's parsing.
struct WithSuppressedFields {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ref_to_uninit")]] int *p1 = &g_uninit;        // OK: rule-targeted suppress
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int *p2 = &g_uninit;                                // OK: whole-profile suppress
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int *p3 [[ref_to_uninit]] = &g_init;                // OK: suppressed (marked target, initialized source)
};

// A suppress on the enclosing record covers its members' initializers.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] WithClassLevelSuppress {
  int *p = &g_uninit; // OK: suppressed by the class-level attribute
};

[[ref_to_uninit]] int *ret_uninit_ptr_ok() { return &g_uninit; }      // OK
[[ref_to_uninit]] int *ret_uninit_ptr_bad() {
  return &g_init; // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}
int *ret_ptr_bad() {
  return &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
int *ret_ptr_ok() { return &g_init; } // OK

[[ref_to_uninit]] int &ret_uninit_ref_ok() { return g_uninit; } // OK
int &ret_ref_bad() {
  return g_uninit; // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}

int *ret_suppressed() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] return &g_uninit; // OK: suppressed
}

template <typename T>
void template_bad() {
  T *p = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_bad<int>' requested here}}
