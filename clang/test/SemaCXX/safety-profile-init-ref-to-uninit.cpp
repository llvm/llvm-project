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

void test_reference_target() {
  int &r1 [[ref_to_uninit]] = g_uninit; // OK
  int &r2 [[ref_to_uninit]] = g_init;   // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int &r3 = g_uninit;                   // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int &r4 = g_init;                     // OK
  int *p [[ref_to_uninit]] = &g_uninit;
  int &r5 [[ref_to_uninit]] = *p;       // OK: *p denotes uninitialized storage
  (void)r1; (void)r2; (void)r3; (void)r4; (void)r5; (void)p;
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

struct WithFields {
  int *p1 [[ref_to_uninit]] = &g_uninit; // OK
  int *p2 [[ref_to_uninit]] = &g_init;   // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *p3 = &g_uninit;                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *p4 = &g_init;                      // OK
  int *p5 = nullptr;                      // OK
  int &r1 [[ref_to_uninit]] = g_uninit;  // OK
  int &r2 = g_uninit;                     // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};

template <typename T>
void template_bad() {
  T *p = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_bad<int>' requested here}}
