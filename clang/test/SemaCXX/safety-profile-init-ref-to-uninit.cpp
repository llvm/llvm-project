// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// std::init / ref_to_uninit (paper §5): a [[ref_to_uninit]] pointer or
// reference must be bound to uninitialized memory, and an unmarked pointer or
// reference must not. This file exercises the rule at variable initialization.

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

int g_init = 0;
int g_init2 = 0;
// Static fixtures supplying uninitialized memory for the pointer/reference
// tests below. A static [[uninit]] is rejected by static_marker (paper section
// 4.2), so suppress that rule here: the test deliberately creates uninitialized
// static storage, and suppression keeps the marker (the source stays
// "uninitialized memory" for the ref_to_uninit checks).
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "static_marker")]] [[uninit]] int g_uninit;
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "static_marker")]] [[uninit]] int g_uninit_arr[3];
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "static_marker")]] [[uninit]] int g_uninit2;
[[ref_to_uninit]] int *allocate(int n);
[[ref_to_uninit]] void *alloc_void();
[[ref_to_uninit]] int &get_uninit_ref();
void h();

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

// A reference cast (an explicit cast yielding a glvalue) denotes the same
// storage as its operand, and a [[ref_to_uninit]]-returning reference call
// denotes uninitialized storage. Symmetric to the pointer side.
void test_reference_casts() {
  int &cr1 [[ref_to_uninit]] = static_cast<int &>(g_uninit); // OK
  int &cr2 = static_cast<int &>(g_uninit);                    // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int &cr3 [[ref_to_uninit]] = (int &)g_init;                 // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}

  int &gr1 [[ref_to_uninit]] = get_uninit_ref(); // OK
  int &gr2 = get_uninit_ref();                    // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  // Address of a reference cast routes back through the pointer recognizer.
  int *p [[ref_to_uninit]] = &(int &)g_uninit; // OK

  (void)cr1; (void)cr2; (void)cr3; (void)gr1; (void)gr2; (void)p;
}

// Pass-through sources are transparent to their operand: a single-element
// braced initializer is looked through to its element, a conditional is
// uninitialized if either arm is, and a comma yields its right operand.
void test_braced_pointer() {
  int *b1 [[ref_to_uninit]] = {&g_uninit}; // OK
  int *b2 = {&g_uninit};                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *b3 [[ref_to_uninit]] = {&g_init};    // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *b4 = {&g_init};                       // OK
  // Empty {} value-initializes to nullptr, like = nullptr: not uninitialized.
  int *b5 [[ref_to_uninit]] = {};            // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *b6 = {};                              // OK
  (void)b1; (void)b2; (void)b3; (void)b4; (void)b5; (void)b6;
}

void test_braced_reference() {
  int &r1 [[ref_to_uninit]] = {g_uninit}; // OK
  int &r2 = {g_uninit};                    // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int &r3 [[ref_to_uninit]] = {g_init};    // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int &r4 = {g_init};                       // OK
  (void)r1; (void)r2; (void)r3; (void)r4;
}

void test_conditional_pointer(bool c) {
  int *p1 = c ? &g_uninit : &g_init;                       // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *p2 [[ref_to_uninit]] = c ? &g_uninit : &g_uninit2;  // OK: both arms uninitialized
  int *p3 [[ref_to_uninit]] = c ? &g_uninit : &g_init;     // OK: either arm may be uninitialized
  int *p4 [[ref_to_uninit]] = c ? &g_init : &g_init2;      // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *p5 = c ? &g_init : &g_init2;                        // OK
  (void)p1; (void)p2; (void)p3; (void)p4; (void)p5;
}

void test_conditional_reference(bool c) {
  int &r1 [[ref_to_uninit]] = c ? g_uninit : g_uninit2; // OK
  int &r2 = c ? g_uninit : g_init;                       // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int &r3 [[ref_to_uninit]] = c ? g_init : g_init2;      // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int &r4 = c ? g_init : g_init2;                        // OK
  (void)r1; (void)r2; (void)r3; (void)r4;
}

void test_comma_pointer() {
  int *p1 = (h(), &g_uninit);                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *p2 [[ref_to_uninit]] = (h(), &g_uninit);  // OK
  int *p3 [[ref_to_uninit]] = (h(), &g_init);    // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *p4 = (h(), &g_init);                      // OK
  (void)p1; (void)p2; (void)p3; (void)p4;
}

void test_comma_reference() {
  int &r1 [[ref_to_uninit]] = (h(), g_uninit); // OK
  int &r2 = (h(), g_uninit);                    // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int &r3 [[ref_to_uninit]] = (h(), g_init);    // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int &r4 = (h(), g_init);                       // OK
  (void)r1; (void)r2; (void)r3; (void)r4;
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

// Assignment through indirection: the assigned-to pointer is reached by
// dereference or subscript, so it cannot carry a local [[ref_to_uninit]]
// marker. It is the default unmarked pointer and must not be bound to
// uninitialized memory, exactly as a directly-named pointer would be.
void test_assignment_indirect() {
  int *p = nullptr;
  int **pp = &p;
  *pp = &g_uninit;   // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  *pp = &g_init;     // OK
  (*pp) = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  *pp = new int;     // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  *pp = new int(0);  // OK

  int *arr[3] = {};
  arr[0] = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  arr[1] = &g_init;   // OK
  (void)pp; (void)arr;
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
  [[uninit]] int a2[3];
  uninitialized_fill(a1, 10); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  uninitialized_fill(a2, 10); // OK

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { take_ptr(&g_uninit); } // OK: suppressed
}

// A defaulted pointer or reference argument is checked against the parameter's
// [[ref_to_uninit]] marking at the call site, like an explicit argument. The
// declarations themselves stay clean; the diagnostic fires at the call.
void def_uninit_ptr(int *p [[ref_to_uninit]] = &g_uninit);
void def_uninit_ptr_bad(int *p [[ref_to_uninit]] = &g_init);
void def_ptr(int *p = &g_init);
void def_ptr_bad(int *p = &g_uninit);
void def_uninit_ref(int &r [[ref_to_uninit]] = g_uninit);
void def_uninit_ref_bad(int &r [[ref_to_uninit]] = g_init);
void def_ref(int &r = g_init);
void def_ref_bad(int &r = g_uninit);

void test_default_arguments() {
  def_uninit_ptr();     // OK
  def_uninit_ptr_bad(); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  def_ptr();            // OK
  def_ptr_bad();        // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  def_uninit_ref();     // OK
  def_uninit_ref_bad(); // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  def_ref();            // OK
  def_ref_bad();        // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  // An explicit argument overrides the default and is checked on its own merits.
  def_ptr_bad(&g_init); // OK

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { def_ptr_bad(); } // OK: suppressed
}

// Pass-through sources reach the recognizer at the call-argument site too. A
// braced scalar pointer argument additionally warns (braces around scalar
// initializer), so the braced cases here use references; the pointer recognizer
// is exercised at this site by the conditional and comma forms.
void test_passthrough_call_arguments(bool c) {
  take_uninit_ref({g_uninit}); // OK
  take_uninit_ref({g_init});   // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  take_ref({g_uninit});        // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  take_ref({g_init});          // OK

  take_uninit_ptr(c ? &g_uninit : &g_init); // OK: either arm may be uninitialized
  take_ptr(c ? &g_uninit : &g_init);        // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  take_uninit_ref(c ? g_uninit : g_uninit2);// OK
  take_ref(c ? g_uninit : g_init);          // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  take_uninit_ptr((h(), &g_uninit)); // OK
  take_ptr((h(), &g_uninit));        // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  take_uninit_ref((h(), g_init));    // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  take_ref((h(), g_uninit));         // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
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

// Pass-through sources at the return site mirror the variable-init behavior. A
// braced scalar pointer return warns (braces around scalar initializer), so the
// braced returns use references; the pointer recognizer is reached here by the
// conditional and comma forms.
[[ref_to_uninit]] int &ret_braced_ref_ok() { return {g_uninit}; } // OK
int &ret_braced_ref_bad() {
  return {g_uninit}; // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
[[ref_to_uninit]] int &ret_braced_ref_bad2() {
  return {g_init}; // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}

[[ref_to_uninit]] int *ret_cond_ptr_ok(bool c) { return c ? &g_uninit : &g_init; } // OK
int *ret_cond_ptr_bad(bool c) {
  return c ? &g_uninit : &g_init; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}

[[ref_to_uninit]] int *ret_comma_ptr_ok() { return (h(), &g_uninit); } // OK
int *ret_comma_ptr_bad() {
  return (h(), &g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
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

// A *non-dependent* pointer bound to uninitialized memory inside a template
// body is diagnosed once, at instantiation, not on the pattern (no double-fire).
template <typename T>
void template_nondependent_bad() {
  int *p = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_nondependent_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_nondependent_bad<int>' requested here}}

// A default-initialized new-expression that leaves a scalar subobject
// indeterminate (e.g. new int, new int[n], paper §1.2 / §4.3) is a source of
// uninitialized free-store memory; new T(...), new T{...}, and a type with a
// user-provided default constructor are initialized.
namespace std { enum class byte : unsigned char {}; }

struct NewAgg { int x; };
struct NewWithCtor { NewWithCtor(); int x; };

void test_new_scalar() {
  int *n1 [[ref_to_uninit]] = new int;      // OK
  int *n2 = new int;                         // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *n3 = new int(5);                      // OK
  int *n4 = new int();                       // OK
  int *n5 = new int{};                       // OK
  int *n6 [[ref_to_uninit]] = new int(5);    // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *n7 [[ref_to_uninit]] = new int();     // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  (void)n1; (void)n2; (void)n3; (void)n4; (void)n5; (void)n6; (void)n7;
}

void test_new_array(int n) {
  int *a1 [[ref_to_uninit]] = new int[10];   // OK
  int *a2 = new int[10];                      // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *a3 [[ref_to_uninit]] = new int[n];     // OK
  int *a4 = new int[n];                       // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)a1; (void)a2; (void)a3; (void)a4;
}

void test_new_class() {
  NewWithCtor *c1 = new NewWithCtor;                    // OK: user-provided default ctor trusted
  NewWithCtor *c2 [[ref_to_uninit]] = new NewWithCtor;  // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  NewAgg *a1 = new NewAgg;                               // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  NewAgg *a2 [[ref_to_uninit]] = new NewAgg;            // OK
  std::byte *b1 = new std::byte;                         // OK: std::byte exemption inherited
  std::byte *b2 [[ref_to_uninit]] = new std::byte;      // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  (void)c1; (void)c2; (void)a1; (void)a2; (void)b1; (void)b2;
}

struct NewInFields {
  int *p1 [[ref_to_uninit]] = new int; // OK
  int *p2 = new int;                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *p3 = new int(5);                 // OK
};

void test_new_assignment() {
  int *p [[ref_to_uninit]] = new int;
  p = new int;    // OK
  p = new int(5); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *q = new int(0);
  q = new int;    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  q = new int(0); // OK
  (void)p; (void)q;
}

void test_new_call_arguments() {
  take_uninit_ptr(new int);    // OK
  take_uninit_ptr(new int(5)); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  take_ptr(new int);           // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  take_ptr(new int(5));        // OK
}

[[ref_to_uninit]] int *ret_new_uninit_ok() { return new int; } // OK
[[ref_to_uninit]] int *ret_new_uninit_bad() {
  return new int(5); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}
int *ret_new_ptr_bad() {
  return new int; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
int *ret_new_ptr_ok() { return new int(0); } // OK

void test_new_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ref_to_uninit")]] int *s = new int; // OK: suppressed
  (void)s;
}

template <typename T>
void template_new_bad() {
  T *p = new T; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_new_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_new_bad<int>' requested here}}

// The call-argument, pointer-assignment, and return sites pass no Decl, so
// (unlike the variable-init site, template_nondependent_bad above) their
// deferral cannot come from D->isTemplated(). They must still fire exactly
// once, at instantiation -- not twice, and not on the pattern.
template <typename T>
void template_call_arg_unmarked() {
  take_ptr(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template void template_call_arg_unmarked<int>(); // expected-note {{in instantiation of function template specialization 'template_call_arg_unmarked<int>' requested here}}

template <typename T>
void template_call_arg_marked() {
  take_uninit_ptr(&g_init); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}
template void template_call_arg_marked<int>(); // expected-note {{in instantiation of function template specialization 'template_call_arg_marked<int>' requested here}}

template <typename T>
void template_assignment_bad() {
  int *p = nullptr;
  p = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_assignment_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_assignment_bad<int>' requested here}}

template <typename T>
int *template_return_bad() {
  return &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template int *template_return_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_return_bad<int>' requested here}}

// A never-instantiated template stays silent: the deferred checks never run.
template <typename T>
void template_never_instantiated() {
  take_ptr(&g_uninit);
  int *p = nullptr;
  p = &g_uninit;
  (void)p;
}

// A violation in a discarded if-constexpr branch is never instantiated, so it
// is not diagnosed -- even though the dependent condition keeps the branch live
// on the pattern (where the check now defers).
template <typename T>
void template_discarded_branch() {
  if constexpr (sizeof(T) > 1000) {
    take_ptr(&g_uninit);
    int *p = nullptr;
    p = &g_uninit;
    (void)p;
  }
}
template void template_discarded_branch<int>();

// std::init / uninit_read (paper §4.5): a read *through* a [[ref_to_uninit]]
// pointer or reference yields an uninitialized value, diagnosed at the
// lvalue-to-rvalue conversion (Sema::DefaultLvalueConversion). These reads fire
// only under -fprofiles; the no-profiles run stays clean. A direct read of a
// named [[uninit]] object is left to the flow-based uninit_read pass and is not
// retested here.
void take_value(int v);

int test_read_through_pointer(int *p [[ref_to_uninit]], int i) {
  int y1 = *p;     // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  int y2 = p[i];   // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  int y3 = *p + 1; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  take_value(*p);  // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  if (*p)          // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
    h();
  (void)y1; (void)y2; (void)y3;
  return *p;       // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
}

void test_read_through_reference(int &r [[ref_to_uninit]], Inner *ptr [[ref_to_uninit]],
                                 void *vp [[ref_to_uninit]]) {
  int y1 = r;                // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  int y2 = ptr->m;           // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  int y3 = (*ptr).m;         // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  int y4 = *(int *)vp;       // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  int y5 = get_uninit_ref(); // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)y1; (void)y2; (void)y3; (void)y4; (void)y5;
}

// Paper §4.5: reading an uninitialized std::byte is permitted, so a read
// through a [[ref_to_uninit]] std::byte pointer/reference is not diagnosed.
void test_read_byte_exempt(std::byte *bp [[ref_to_uninit]], std::byte &br [[ref_to_uninit]]) {
  std::byte b1 = *bp; // OK
  std::byte b2 = br;  // OK
  (void)b1; (void)b2;
}

void test_read_suppress(int *p [[ref_to_uninit]]) {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { take_value(*p); }                      // OK: whole-profile suppress
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] { take_value(*p); } // OK: rule-targeted suppress
}

// None of these is a read through the marker: a discarded-value expression and
// an address-of apply no lvalue-to-rvalue conversion, a write targets the
// glvalue without loading it, a reference binding is not a load, and copying
// the pointer value reads the (initialized) pointer object rather than through
// it.
void test_read_negatives(int *p [[ref_to_uninit]], int &r [[ref_to_uninit]],
                         Inner *ptr [[ref_to_uninit]], int *base [[ref_to_uninit]]) {
  (void)r;                         // OK: discarded value
  (void)*p;                        // OK: discarded value
  int *ap [[ref_to_uninit]] = &*p; // OK: address-of is not a read
  *p = 5;                          // OK: write, not a read
  ptr->m = 5;                      // OK: write, not a read
  int &r2 [[ref_to_uninit]] = *p;  // OK: reference binding, not a read
  int *q [[ref_to_uninit]] = base; // OK: reads the pointer value, not through it
  (void)ap; (void)q; (void)r2;
}

// A read through a [[ref_to_uninit]] parameter inside a template body defers on
// the pattern (a dependent context) and fires once, at instantiation -- whether
// the read's operand is non-dependent (template_read_nondependent_bad) or
// dependent (template_read_dependent_bad). A never-instantiated template stays
// silent. Mirrors the binding template_* cases above.
template <typename T>
void template_read_nondependent_bad(int *p [[ref_to_uninit]]) {
  int y = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)y;
}
template void template_read_nondependent_bad<int>(int *); // expected-note {{in instantiation of function template specialization 'template_read_nondependent_bad<int>' requested here}}

template <typename T>
T template_read_dependent_bad(T *p [[ref_to_uninit]]) {
  return *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
}
template int template_read_dependent_bad<int>(int *); // expected-note {{in instantiation of function template specialization 'template_read_dependent_bad<int>' requested here}}

template <typename T>
void template_read_never_instantiated(int *p [[ref_to_uninit]]) {
  int y = *p;
  (void)y;
}
