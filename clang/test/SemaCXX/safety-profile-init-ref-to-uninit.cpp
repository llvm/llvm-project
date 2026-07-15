// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -fcxx-exceptions -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -fcxx-exceptions -std=c++23 %s

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
  // Empty {} value-initializes to nullptr, like = nullptr: a null source is
  // consistent with marked and unmarked targets alike (paper §8, §4.3; see
  // test_null_sources).
  int *b5 [[ref_to_uninit]] = {};            // OK
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

// A null pointer refers to no object, so it is consistent with a marked
// target -- the marker means "zero or more uninitialized objects" (paper §8)
// -- and with an unmarked one (the paper §4.3 f1(p2) example): it classifies
// as unknown storage. This covers the literal forms and a named local whose
// declaration initializer is null.
void test_null_sources() {
  take_uninit_ptr(nullptr); // OK: null literal for a marked param
  take_uninit_ptr(0);       // OK: literal zero
  take_ptr(nullptr);        // OK

  int *p2 = nullptr;
  take_uninit_ptr(p2); // OK: the paper §4.3 example
  take_ptr(p2);        // OK

  int *q [[ref_to_uninit]] = p2; // OK: null is not affirmatively initialized
  int *q2 = p2;                  // OK
  (void)q; (void)q2;

  int *z = {};            // value-initializes to null, like = nullptr
  take_uninit_ptr(z);     // OK
  int *zb [[ref_to_uninit]] = {nullptr}; // OK: braced null recurses to the literal
  (void)zb;

  // The null-init classification is parse-order lenient: a reassignment after
  // the null declaration is not tracked, so passing the now-initialized
  // pointer to a marked parameter is an accepted missed diagnostic.
  int *r = nullptr;
  r = &g_init;
  take_uninit_ptr(r); // accepted: missed diagnostic (parse-order leniency)
}

// A zero-initialized *global* null pointer stays classified initialized:
// an extern pointer may be initialized elsewhere (another translation unit),
// and keeping globals initialized preserves the marked-direction
// diagnostics. Deliberate residual strictness.
int *g_null_ptr;
void test_null_global() {
  take_uninit_ptr(g_null_ptr); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}

// A static local is excluded for the same reason as a global (not
// function-local state; hasLocalStorage is the gate).
void test_null_static_local() {
  static int *sp = nullptr;
  take_uninit_ptr(sp); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}

// A *parameter* is excluded too: a ParmVarDecl's initializer is its default
// argument, which is not the parameter's value on most calls -- a
// defaulted-null parameter may be passed any caller pointer, so it must keep
// drawing the marked-target diagnostic.
void null_default_param(int *cp = nullptr) {
  take_uninit_ptr(cp); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}

// At the *call site* an omitted defaulted argument is the null literal
// itself: fine for a marked parameter.
void null_default_marked(int *p [[ref_to_uninit]] = nullptr);
void test_null_default_marked_param() {
  null_default_marked();        // OK: the null default argument
  null_default_marked(nullptr); // OK
}

// A *marked* pointer initialized to null keeps its marker classification as
// a source: the explicit marker is respected over the null initializer.
void test_null_init_marked_decl() {
  int *m [[ref_to_uninit]] = nullptr;
  int *m2 = m; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)m2;
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

// Call arguments are checked at parameter copy-initialization, which also
// covers call forms that never reach GatherArgumentsForCall: calls to objects
// of class type (functors, lambdas) and overloaded operators (member and
// non-member).
struct MarkedFunctor {
  void operator()(int *p [[ref_to_uninit]]);
};
struct UnmarkedFunctor {
  void operator()(int *p);
};

void test_functor_arguments() {
  MarkedFunctor mf;
  mf(&g_uninit); // OK
  mf(&g_init);   // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  UnmarkedFunctor uf;
  uf(&g_init);   // OK
  uf(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { uf(&g_uninit); } // OK: suppressed
}

void test_lambda_arguments() {
  auto l = [](int *p [[ref_to_uninit]]) {};
  l(&g_uninit); // OK
  l(&g_init);   // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}

// A call with no declared callee (through a function pointer) has no
// parameter declaration that could carry [[ref_to_uninit]], so its arguments
// are checked as unmarked targets (paper §7.2: passing uninitialized memory
// needs an appropriately declared callee). That holds even when the pointer
// happens to point at a function whose parameter is marked -- the marker is a
// declaration property, invisible through the pointer (paper §1.3, local
// analysis); suppress at the call if the flow is intended.
void test_fnptr_call_arguments(void (*fp)(int *), void (*fr)(int &)) {
  fp(&g_init);   // OK
  fp(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *rtu [[ref_to_uninit]] = &g_uninit;
  fp(rtu);       // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  fr(g_init);    // OK
  fr(g_uninit);  // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  void (*marked)(int *) = take_uninit_ptr;
  marked(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { fp(&g_uninit); } // OK: suppressed
  (void)rtu;
}

// Decl-less like the declared-callee argument site; the dependent callee type
// keeps the call unchecked on the pattern, so it fires once, at instantiation.
template <typename T>
void template_fnptr_call_arg(void (*fp)(T *)) {
  fp(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template void template_fnptr_call_arg<int>(void (*)(int *)); // expected-note {{in instantiation of function template specialization 'template_fnptr_call_arg<int>' requested here}}

// A variadic (...) argument never reaches parameter copy-initialization, and
// a ... parameter cannot carry [[ref_to_uninit]], so a pointer passed through
// it is checked as an unmarked target (paper §7.2). A *value* passed through
// ... is promoted with an ordinary lvalue-to-rvalue load, so its read is the
// read-through chokepoint's, not this site's.
void vf(int, ...);

void test_variadic_arguments() {
  vf(0, &g_init);      // OK
  vf(0, &g_uninit);    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *rtu [[ref_to_uninit]] = &g_uninit;
  vf(0, rtu);          // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  vf(0, g_uninit_arr); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  vf(0, *rtu);         // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}

  // Through a variadic function pointer: the named arguments are the
  // no-declared-callee site's, the ... arguments this one's -- exactly one
  // diagnostic either way.
  void (*vfp)(int, ...) = vf;
  vfp(0, &g_uninit);   // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { vf(0, &g_uninit); } // OK: suppressed
  (void)rtu;
}

// Decl-less with a non-dependent argument: fires at definition time, and
// again when the call (always rebuilt) re-promotes the argument at
// instantiation -- the accepted repetition.
template <typename T>
void template_variadic_arg() {
  vf(0, &g_uninit); // expected-error 2 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template void template_variadic_arg<int>(); // expected-note {{in instantiation of function template specialization 'template_variadic_arg<int>' requested here}}

// A call to an object of class type promotes its variadic arguments in its
// own loop (Sema::BuildCallToObjectOfClassType), distinct from
// GatherArgumentsForCall's; both are hooked, so variadic functors and
// variadic lambdas are covered too.
struct VariadicFunctor {
  void operator()(int, ...);
};

void test_variadic_functor_arguments() {
  VariadicFunctor f;
  f(0, &g_init);   // OK
  f(0, &g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  auto l = [](int, ...) {};
  l(0, &g_init);   // OK
  l(0, &g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}

// A surrogate call converts to a function pointer and calls through it, so
// its named arguments are the no-declared-callee site's.
struct Surrogate {
  using FP = void (*)(int *);
  operator FP();
};

void test_surrogate_call_arguments() {
  Surrogate s;
  s(&g_init);   // OK
  s(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}

// An init-capture is a binding: a capture cannot carry [[ref_to_uninit]], so
// capturing a pointer or reference to uninitialized memory is always the
// unmarked-direction violation.
void test_init_captures() {
  auto c1 = [p = &g_init] { (void)p; };   // OK
  auto c2 = [p = &g_uninit] { (void)p; }; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *rtu [[ref_to_uninit]] = &g_uninit;
  auto c3 = [&r = *rtu] { (void)r; }; // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  auto c4 = [q = rtu] { (void)q; };   // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  auto c5 = [v = g_init] { (void)v; }; // OK: a by-value int copy is not a binding
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] {
    auto c6 = [p = &g_uninit] { (void)p; }; // OK: suppressed
    (void)c6;
  }
  (void)c1; (void)c2; (void)c3; (void)c4; (void)c5;
}

// An init-capture inside a template body defers on the pattern and fires
// once, at instantiation.
template <typename T>
void template_init_capture_bad() {
  auto c = [p = &g_uninit] { (void)p; }; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)c;
}
template void template_init_capture_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_init_capture_bad<int>' requested here}}

// A by-reference capture -- explicit or via a capture-default -- is the same
// binding as an init-capture: a capture cannot carry [[ref_to_uninit]], so
// capturing an [[uninit]] variable (or a [[ref_to_uninit]] reference) by
// reference is always the unmarked-direction violation. A copy capture is not
// a binding; it reads the variable in the enclosing function's CFG, which is
// the flow-based uninit_read pass's territory.
void test_ref_captures() {
  int x [[uninit]];
  int ok = 0;
  // The bodies must not assign x: a body store would credit it in parse
  // order and silence the capture check (see test_ref_capture_body_store).
  auto c1 = [&x] { (void)x; }; // expected-error {{capturing 'x' by reference binds a reference to uninitialized memory under profile 'std::init'}}
  auto c2 = [&] { (void)x; };  // expected-error {{capturing 'x' by reference binds a reference to uninitialized memory under profile 'std::init'}}
  auto c3 = [&ok] { ok = 1; }; // OK: initialized
  int *rtu [[ref_to_uninit]] = &g_uninit;
  auto c4 = [&rtu] { (void)rtu; }; // OK: the pointer object itself is initialized
  int &ur [[ref_to_uninit]] = *rtu;
  auto c5 = [&ur] { (void)ur; }; // expected-error {{capturing 'ur' by reference binds a reference to uninitialized memory under profile 'std::init'}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ref_to_uninit")]] {
    auto c6 = [&x] { (void)x; }; // OK: suppressed
    (void)c6;
  }
  (void)c1; (void)c2; (void)c3; (void)c4; (void)c5;
}

// Parse-order store credit reaches the capture check both ways -- accepted
// leniencies of the scope-less credit map (false negatives only):
void test_ref_capture_after_store() {
  int u [[uninit]];
  u = 5;
  auto c = [&u] { (void)u; }; // OK: the store initialized u (symmetric
                              // with binding &u after the store)
  (void)c;
}

void test_ref_capture_body_store() {
  int u [[uninit]];
  auto L = [&] { u = 5; }; // OK: the body's own store credits u at parse
                           // order, silencing this capture check -- the
                           // deliberate leniency (no FunctionScopeInfo
                           // scoping), pinned here
  int *q = &u;             // OK: credited by the body store above
  (void)L; (void)q;
}

// A by-reference capture of a variable with a non-dependent type fires at
// definition time, and again when TreeTransform's unconditional lambda
// rebuild re-processes the capture at instantiation -- the accepted
// repetition. (The body must not assign x, as in test_ref_captures.)
template <typename T>
void template_ref_capture_bad() {
  int x [[uninit]];
  auto c = [&x] { (void)x; }; // expected-error 2 {{capturing 'x' by reference binds a reference to uninitialized memory under profile 'std::init'}}
  (void)c;
}
template void template_ref_capture_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_ref_capture_bad<int>' requested here}}

struct OpTag {};
OpTag operator+(OpTag, int *p [[ref_to_uninit]]);

struct Assignable {
  Assignable &operator=(int *p [[ref_to_uninit]]);
};

void test_operator_arguments() {
  OpTag t;
  (void)(t + &g_uninit); // OK
  (void)(t + &g_init);   // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}

  Assignable a;
  a = &g_uninit; // OK
  a = &g_init;   // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}

// A non-dependent functor-call argument fires at definition time like every
// other Decl-less binding site, and repeats when the call is rebuilt at
// instantiation (the local functor forces the rebuild).
template <typename T>
void template_functor_bad() {
  MarkedFunctor mf;
  mf(&g_init); // expected-error 2 {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}
template void template_functor_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_functor_bad<int>' requested here}}

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

// A dependent [[ref_to_uninit]] parameter defers marker validation to
// instantiation (the pattern is accepted); the instantiated parameter carries
// the marker and drives this rule at the call.
template <typename T>
void dependent_marked_fill(T p [[ref_to_uninit]]) { (void)p; }
void test_dependent_marked_param() {
  dependent_marked_fill<int *>(&g_uninit); // OK: marked target, uninit source
  dependent_marked_fill<int *>(&g_init); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}

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

// A written initializer for an *allocated pointer* is itself a binding: the
// heap pointer object cannot carry [[ref_to_uninit]], so it must not be bound
// to uninitialized memory. Both the parenthesized and the braced form are
// checked; copying the value of a marked pointer is the same violation, as at
// variable scope.
void test_new_pointer_init() {
  int **n1 = new (int *)(&g_init);   // OK
  int **n2 = new (int *)(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int **n3 = new (int *){&g_uninit}; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *rtu [[ref_to_uninit]] = &g_uninit;
  int **n4 = new (int *)(rtu); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int **n5 [[ref_to_uninit]] = new (int *); // OK: no written initializer -- the
                                            // allocated pointer is indeterminate,
                                            // so the marked target accepts it
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] {
    int **s = new (int *)(&g_uninit); // OK: suppressed
    (void)s;
  }
  (void)n1; (void)n2; (void)n3; (void)n4; (void)n5;
}

// A dependent allocated type defers on the pattern and fires once, at
// instantiation.
template <typename T>
void template_new_pointer_bad() {
  T **p = new (T *)(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_new_pointer_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_new_pointer_bad<int>' requested here}}

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
// (unlike the variable-init site, template_nondependent_bad above) they defer
// only on an instantiation-dependent source. These sources are non-dependent,
// so each fires at definition time -- and each construct is rebuilt at
// instantiation anyway (the callee's implicit cast is stripped, forcing a
// call rebuild; the local p is remapped; a return statement always rebuilds),
// so the diagnostic repeats there. The repetition is accepted for now.
template <typename T>
void template_call_arg_unmarked() {
  take_ptr(&g_uninit); // expected-error 2 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template void template_call_arg_unmarked<int>(); // expected-note {{in instantiation of function template specialization 'template_call_arg_unmarked<int>' requested here}}

template <typename T>
void template_call_arg_marked() {
  take_uninit_ptr(&g_init); // expected-error 2 {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
}
template void template_call_arg_marked<int>(); // expected-note {{in instantiation of function template specialization 'template_call_arg_marked<int>' requested here}}

template <typename T>
void template_assignment_bad() {
  int *p = nullptr;
  p = &g_uninit; // expected-error 2 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_assignment_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_assignment_bad<int>' requested here}}

template <typename T>
int *template_return_bad() {
  return &g_uninit; // expected-error 2 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template int *template_return_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_return_bad<int>' requested here}}

// The definition-time fire repeats once per instantiation that rebuilds the
// construct: two explicit instantiations pin the exact counts (one pattern
// fire plus one per specialization).
template <typename T>
void template_assignment_repeats() {
  int *p = nullptr;
  p = &g_uninit; // expected-error 3 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_assignment_repeats<int>();  // expected-note {{in instantiation of function template specialization 'template_assignment_repeats<int>' requested here}}
template void template_assignment_repeats<long>(); // expected-note {{in instantiation of function template specialization 'template_assignment_repeats<long>' requested here}}

template <typename T>
int *template_return_repeats() {
  return &g_uninit; // expected-error 3 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template int *template_return_repeats<int>();  // expected-note {{in instantiation of function template specialization 'template_return_repeats<int>' requested here}}
template int *template_return_repeats<long>(); // expected-note {{in instantiation of function template specialization 'template_return_repeats<long>' requested here}}

// An instantiation-dependent source is not checkable on the pattern: no
// definition-time fire. The construct is rebuilt at every instantiation, so
// each violating specialization diagnoses once, with its note chain.
template <typename T>
void template_dependent_per_spec() {
  T *p = nullptr;
  p = (T *)&g_uninit; // expected-error 2 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}
template void template_dependent_per_spec<int>();  // expected-note {{in instantiation of function template specialization 'template_dependent_per_spec<int>' requested here}}
template void template_dependent_per_spec<long>(); // expected-note {{in instantiation of function template specialization 'template_dependent_per_spec<long>' requested here}}

// Fully non-dependent constructs whose operands transform to themselves are
// *reused* by TreeTransform at instantiation -- their Build* never re-runs.
// Deferring would silently lose the diagnostic (these all-global shapes were
// silent before), so they are checked at definition time: exactly one error,
// on the pattern, with no instantiation note.
int *g_ptr_sink = nullptr;
int **g_pp_sink = nullptr;

template <typename T>
void template_allglobal_bad() {
  g_ptr_sink = &g_uninit;             // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  throw &g_uninit;                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  g_pp_sink = new (int *)(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template void template_allglobal_bad<int>();

// The same shapes diagnose in a never-instantiated template: definition-time
// checking deliberately trades strict "as-if after phase 7" purity for
// reuse-proof diagnostics.
template <typename T>
void template_allglobal_never_instantiated() {
  g_ptr_sink = &g_uninit;             // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  throw &g_uninit;                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  g_pp_sink = new (int *)(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}

// A never-instantiated template diagnoses its non-dependent violations at
// definition time; only instantiation-dependent constructs (the (T *) cast)
// stay silent without an instantiation.
template <typename T>
void template_never_instantiated() {
  take_ptr(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *p = nullptr;
  p = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  p = (T *)&g_uninit;
  (void)p;
}

// A value-dependent if-constexpr condition is not yet known discarded at the
// pattern, so the branch is live at parse and its non-dependent violations
// diagnose at definition time. The f<int> instantiation discards the branch
// (never rebuilding its statements), so nothing repeats.
template <typename T>
void template_discarded_branch() {
  if constexpr (sizeof(T) > 1000) {
    take_ptr(&g_uninit); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
    int *p = nullptr;
    p = &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
    (void)p;
  }
}
template void template_discarded_branch<int>();

// A generic lambda's body is a template pattern even in a non-template
// function: a non-dependent violation diagnoses at definition time whether or
// not the lambda is ever invoked, and an all-global shape is reused (not
// rebuilt) when the call operator is instantiated, so invoking does not
// repeat it.
void generic_lambda_never_invoked() {
  auto l = [](auto x) { g_ptr_sink = &g_uninit; }; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)l;
}

void generic_lambda_invoked() {
  auto l = [](auto x) { g_ptr_sink = &g_uninit; }; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  l(1);
}

// A late-parsed inline member of a class template is a pattern too: a
// non-dependent violation diagnoses when its body is parsed. Suppression on
// the method or on the class covers the definition-time fire like any other.
template <typename T>
struct LateParsedMember {
  void m() { g_ptr_sink = &g_uninit; } // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};

template <typename T>
struct LateParsedSuppressMethod {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] void m() { g_ptr_sink = &g_uninit; } // OK: suppressed
};

template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] LateParsedSuppressClass {
  void m() { g_ptr_sink = &g_uninit; } // OK: suppressed
};

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

// The suppression dominion of a member declaration includes its initializer
// tokens (P3589R2 s2.4p3). The read-through check fires from
// ActOnFinishCXXInClassMemberInitializer during the late parse of an NSDMI,
// so a suppression on the field (or, via the lexical parent walk, on the
// class) must cover it; a sibling member's suppression must not leak.
int *nsdmi_rtu [[ref_to_uninit]] = &g_uninit;

struct NsdmiReadSuppress {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int x = *nsdmi_rtu; // OK: field suppress
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] int y = *nsdmi_rtu; // OK: rule-targeted
  int z = *nsdmi_rtu; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
};

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] NsdmiClassReadSuppress {
  int x = *nsdmi_rtu; // OK: class-level suppression via the lexical parent walk
};

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
  int &r2 [[ref_to_uninit]] = *p;  // OK: reference binding, not a read
  *p = 5;                          // OK: write, not a read (and it credits
                                   // p's pointee, so it stays after the
                                   // marked bindings above)
  ptr->m = 5;                      // OK: write, not a read
  int *q [[ref_to_uninit]] = base; // OK: reads the pointer value, not through it
  (void)ap; (void)q; (void)r2;
}

// A subobject read of a named [[uninit]] object loads an uninitialized value,
// exactly like a read through a [[ref_to_uninit]] pointer: member-wise delayed
// initialization of an [[uninit]] object is banned (paper §5.4), so no
// assignment could have given the member a value. Only the *whole-object*
// direct read of a named [[uninit]] entity is left to the flow-based
// uninit_read pass (which credits assignments).
struct Pair { int x; int y; };
struct PairHolder { Pair p; };

void test_member_read_of_uninit_object() {
  Pair s [[uninit]];
  int y1 = s.x;    // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  take_value(s.y); // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  PairHolder o [[uninit]];
  int y2 = o.p.x;  // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  // The arrow spelling reaches the member through a pointer, so the
  // diagnostic's phrasing approximation picks the pointer wording; the read is
  // diagnosed all the same.
  int y3 = (&s)->x; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)y1; (void)y2; (void)y3;
}

// Discarded values and address-taking apply no lvalue-to-rvalue conversion
// and are not reads; taking the member's address is the binding checks'
// territory (unchanged behavior, retested as a regression guard). A write is
// not a read either, but a subobject store of an [[uninit]] object is itself
// banned as delayed initialization (uninit_write; full coverage in
// safety-profile-init-write.cpp).
void test_member_read_negatives() {
  Pair s [[uninit]];
  s.x = 1;                         // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  (void)s.x;                       // OK: discarded value
  int *p = &s.x;                   // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int *q [[ref_to_uninit]] = &s.x; // OK: binding checked by ref_to_uninit
  (void)p; (void)q;
}

void test_member_read_suppress() {
  Pair s [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] { take_value(s.x); } // OK
}

// std::byte members stay exempt (paper §4.5).
struct WithByte { std::byte b; int i; };
void test_member_read_byte_exempt() {
  WithByte s [[uninit]];
  std::byte b = s.b; // OK
  (void)b;
}

// A whole-record copy from *pp reads the uninitialized pointee (paper
// abstract: an object marked [[ref_to_uninit]] cannot be read through), but
// class types never reach the lvalue-to-rvalue chokepoint. The copy is caught
// all the same, by the binding rule at the copy constructor's reference
// parameter -- so the diagnostic is the binding one, not the read-through
// one. This holds for copy-, direct-, braced-, argument-, and return-copies
// alike.
void take_pair(Pair v);

Pair test_record_copy_through(Pair *pp [[ref_to_uninit]]) {
  Pair v = *pp;   // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  Pair w(*pp);    // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  Pair b = {*pp}; // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  take_pair(*pp); // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)v; (void)w; (void)b;
  return *pp;     // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}

// The escape is the paper's own (§7.2): a copy constructor declared with a
// [[ref_to_uninit]] parameter accepts the uninitialized source -- and then,
// symmetrically, rejects an initialized one.
struct MarkedCopy {
  int x;
  MarkedCopy();
  MarkedCopy(const MarkedCopy &q [[ref_to_uninit]]);
};

void test_record_copy_marked_ctor(MarkedCopy *mp [[ref_to_uninit]],
                                  const MarkedCopy &init) {
  MarkedCopy v = *mp;  // OK: the marked parameter accepts the uninit pointee
  MarkedCopy w = init; // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  (void)v; (void)w;
}

// A record *containing* std::byte members is not std::byte, so the §4.5 read
// exemption does not extend to the whole-record binding.
struct ByteBox { std::byte b; };
void test_record_copy_byte_member(ByteBox *bp [[ref_to_uninit]]) {
  ByteBox v = *bp; // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)v;
}

// An element read of a named [[uninit]] array is a subobject read exactly like
// s.x: neither flow pass tracks array elements (and element-wise delayed
// initialization is banned, paper §5.5), so the marker counts below the
// element access even for a read. *a denotes the same element as a[0].
void test_element_read_of_uninit_array(int i) {
  [[uninit]] int a[2];
  int y1 = a[0];    // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  take_value(a[i]); // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  int y2 = *a;      // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  [[uninit]] int m[2][2];
  int y3 = m[1][0]; // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  (void)y1; (void)y2; (void)y3;
}

// An [[uninit]] array *member*'s element read is flagged the same way: like
// the class-type member in HasAggMember below, an array member has no legal
// element-wise assignment path, so the marker counts even on the current
// object.
struct WithArrMember {
  [[uninit]] int a[2];
  int get() { return a[0]; } // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
};

// Discarded values and address-taking apply no lvalue-to-rvalue conversion
// and are not reads; bindings to the array or its elements stay the
// ref_to_uninit checks' territory (regression guards, unchanged behavior). An
// element store is not a read either, but is itself banned as delayed
// initialization (uninit_write; full coverage in
// safety-profile-init-write.cpp).
void test_element_read_negatives(int i) {
  [[uninit]] int a[2];
  a[0] = 1;                         // expected-error {{writing an element of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  (void)a[0];                       // OK: discarded value
  int *p [[ref_to_uninit]] = &a[0]; // OK: address-of is not a read; binding checked by ref_to_uninit
  int *q [[ref_to_uninit]] = a;     // OK: array decay is a binding, not a read
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] { take_value(a[i]); } // OK: rule-targeted suppress
  (void)p; (void)q;
}

// std::byte arrays stay exempt (paper §4.5).
void test_element_read_byte_exempt() {
  [[uninit]] std::byte b[2];
  std::byte v = b[0]; // OK
  (void)v;
}

// The dependent element type makes the read instantiation-dependent, so it
// defers on the pattern and fires once, at instantiation.
template <typename T>
void template_element_read_bad() {
  [[uninit]] T a[2];
  int y = a[0]; // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  (void)y;
}
template void template_element_read_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_element_read_bad<int>' requested here}}

// The §5.2 trust pattern is preserved: a scalar [[uninit]] *member* of the
// current object may be assigned in the constructor body (flow-checked by the
// ctor-body pass), so a member-function read of it is not flagged here.
struct BodyInit {
  int m [[uninit]];
  BodyInit() { m = 1; }
  int get() { return m; } // OK: trusted, assigned in the constructor body
};

// But a subobject of an [[uninit]] *class-type member* has no legal
// assignment path (member-wise delayed initialization is banned, and
// construct_at flow is uniformly unmodeled), so its read is flagged even on
// the current object.
struct HasAggMember {
  Pair agg [[uninit]];
  int get() { return agg.x; } // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
};

// Like every Decl-less check, the member read fires at definition time when
// its glvalue is non-dependent, and repeats when the read is rebuilt at
// instantiation (the local s is remapped) -- the accepted repetition.
template <typename T>
void template_member_read_bad() {
  Pair s [[uninit]];
  int y = s.x; // expected-error 2 {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  (void)y;
}
template void template_member_read_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_member_read_bad<int>' requested here}}

// A read through a [[ref_to_uninit]] parameter inside a template body fires at
// definition time when the operand is non-dependent
// (template_read_nondependent_bad; the parameter remap rebuilds the read at
// instantiation, repeating the diagnostic) and defers to instantiation when it
// is dependent (template_read_dependent_bad). Mirrors the binding template_*
// cases above.
template <typename T>
void template_read_nondependent_bad(int *p [[ref_to_uninit]]) {
  int y = *p; // expected-error 2 {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)y;
}
template void template_read_nondependent_bad<int>(int *); // expected-note {{in instantiation of function template specialization 'template_read_nondependent_bad<int>' requested here}}

template <typename T>
T template_read_dependent_bad(T *p [[ref_to_uninit]]) {
  return *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
}
template int template_read_dependent_bad<int>(int *); // expected-note {{in instantiation of function template specialization 'template_read_dependent_bad<int>' requested here}}

// A never-instantiated pattern diagnoses its non-dependent read at definition
// time, exactly once.
template <typename T>
void template_read_never_instantiated(int *p [[ref_to_uninit]]) {
  int y = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)y;
}

// An all-global read: the definition-time fire, plus a repeat when the
// initialization of the local y is rebuilt at instantiation.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "static_marker")]] [[uninit]] Pair g_uninit_pair;

template <typename T>
void template_global_read_bad() {
  int y = g_uninit_pair.x; // expected-error 2 {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  (void)y;
}
template void template_global_read_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_global_read_bad<int>' requested here}}

template <typename T>
void template_global_read_never_instantiated() {
  int y = g_uninit_pair.x; // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  (void)y;
}

// A compound assignment and a built-in ++/-- read the old value before
// storing, but build no lvalue-to-rvalue node for the operand, so the
// operator sites check the load directly. Every compound form through a
// [[ref_to_uninit]] pointer or reference is diagnosed; the shift forms load
// through their LHS promotion instead and must fire exactly once. Reading an
// unmarked pointer's pointee is trusted, and ++ on the marked pointer itself
// reads the (initialized) pointer object, not through it. Each form gets a
// fresh marker: a compound form both reads (the error) and stores, and the
// store credits the pointee for everything after it in parse order (see
// test_pointee_store_credit) -- except through an element access, which
// never sees the credit (p[i] below, after *p's store; paper §5.4).
void test_compound_read_through(int *p [[ref_to_uninit]], int *q,
                                int &r [[ref_to_uninit]],
                                Inner *ptr [[ref_to_uninit]], int i,
                                int *p2 [[ref_to_uninit]],
                                int *p3 [[ref_to_uninit]],
                                int &r2 [[ref_to_uninit]],
                                int *p4 [[ref_to_uninit]]) {
  *p += 1;     // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  p[i] -= 1;   // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  r *= 2;      // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  ptr->m |= 1; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  ++*p2;       // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (*p3)--;     // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  ++r2;        // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  *p4 <<= 1;   // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  *q += 1;     // OK: an unmarked pointer is trusted initialized
  ++p;         // OK: reads the pointer object itself, not through it
}

void test_compound_read_suppress(int *p [[ref_to_uninit]]) {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] { *p += 1; } // OK: rule-targeted suppress
}

// Like every Decl-less read check, the compound read fires at definition time
// on a non-dependent operand and repeats when the parameter remap rebuilds the
// assignment at instantiation.
template <typename T>
void template_compound_read_bad(int *p [[ref_to_uninit]]) {
  *p += 1; // expected-error 2 {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
}
template void template_compound_read_bad<int>(int *); // expected-note {{in instantiation of function template specialization 'template_compound_read_bad<int>' requested here}}

// Parse-order pointee store credit (paper §4.3/§4.5): a whole-`*p` store
// through a [[ref_to_uninit]] pointer is the pointee's initialization, so
// whole-`*p` accesses after it (in parse order) are legal -- and the paper's
// reverse direction applies: the credited pointer now refers to initialized
// memory and REQUIRES an unmarked target. Element accesses never see the
// credit in either direction (§5.4's random-access ban), and reseating the
// pointer clears it.
void test_pointee_store_credit(int *p [[ref_to_uninit]]) {
  *p = 5;      // OK: the write initializes the pointee (and credits it)
  *p = 7;      // OK: further whole-entity stores stay legal
  int x = *p;  // OK: credited (rejected before the store)
  int *r2 [[ref_to_uninit]] = p; // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  (void)x; (void)r2;
}

void test_pointee_read_before_store(int *p [[ref_to_uninit]]) {
  int x = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  *p = 5;
  (void)x;
}

// The credit is recorded at the tail of the assignment, after the RHS is
// checked: a self-assignment's RHS read must not be silenced by its own
// store (the key recording-order regression test).
void test_pointee_no_self_credit(int *p [[ref_to_uninit]]) {
  *p = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
}

// Element stores neither credit nor invalidate (§5.4/§5.5: element-wise
// state is untrackable by design)...
void test_subscript_store_no_credit(int *p [[ref_to_uninit]]) {
  p[0] = 1;
  int x = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)x;
}

// ...and element reads never see pointee credit: `*p = 5;` must not legalize
// p[1] (the pointee may be an array with only element 0 written). The model
// is purely syntactic, so even p[0] -- the same storage as *p -- stays an
// error: only the whole-`*p` form is credited.
void test_subscript_read_not_credited(int *p [[ref_to_uninit]], int i) {
  *p = 5;
  int x = p[i]; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  int y = p[0]; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)x; (void)y;
}

// Reseating the pointer -- plain assignment, compound arithmetic, or ++ --
// clears its pointee credit: the credit described the old pointee.
void test_reseat_clears_credit(int *p [[ref_to_uninit]],
                               int *q [[ref_to_uninit]], int n) {
  *p = 5;
  p = q;
  int x = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  *p = 5;
  p += n;
  int y = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  *p = 5;
  p++;
  int z = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)x; (void)y; (void)z;
}

// A store through a marked *reference* credits its referent; a reference
// cannot be reseated, so the credit is never cleared.
void test_marked_ref_store_credit(int &r [[ref_to_uninit]]) {
  r = 5;
  int x = r; // OK: the store initialized the referent
  (void)x;
}

void test_marked_ref_read_before_store(int &r [[ref_to_uninit]]) {
  int x = r; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  r = 5;
  (void)x;
}

// Store credit is recorded at pattern-parse time too: non-dependent
// store-then-read inside a template is checked at definition time (the
// documented phase-7 trade-off) and must find the pattern-time credit;
// instantiations rebuild every DeclRefExpr against fresh declarations and
// re-record independently.
template <typename T>
void template_store_then_read(int *p [[ref_to_uninit]]) {
  *p = 5;
  int x = *p; // OK at definition time and at instantiation
  (void)x;
}
template void template_store_then_read<int>(int *);

// A store in a discarded if-constexpr branch is not instantiated, so the
// rebuilt read finds no credit at that instantiation -- while the pattern's
// store did credit the definition-time check (a dependent condition
// discards nothing at parse).
template <bool B>
void template_discarded_store(int *p [[ref_to_uninit]]) {
  if constexpr (B)
    *p = 5;
  int x = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)x;
}
template void template_discarded_store<true>(int *);  // OK: store instantiated
template void template_discarded_store<false>(int *); // expected-note {{in instantiation of function template specialization 'template_discarded_store<false>' requested here}}

// Known residual gap (documented definition-time-purity trade-off): a store
// with a *type-dependent RHS* routes through the overloaded-operator path at
// pattern time and never reaches the built-in assignment funnel, so it earns
// no pattern-time credit and the following non-dependent read
// false-positives at definition time. The instantiation is clean (its
// rebuilt store re-records first) -- exactly one error total.
template <typename T>
void template_dependent_rhs_store(int *p [[ref_to_uninit]], T t) {
  *p = t;
  int x = *p; // expected-error {{read through a '[[ref_to_uninit]]' pointer or reference accesses uninitialized memory under profile 'std::init'}}
  (void)x;
}
template void template_dependent_rhs_store<int>(int *, int);

// std::init / ref_to_uninit (paper §5): a pointer/reference member given a
// *written* constructor member-initializer is checked with the enclosing
// constructor as the Decl, so a class-template pattern defers and fires once at
// instantiation, mirroring ctor_uninit_member. Both the parenthesized and the
// braced member-initializer forms reach the same recognizer.
struct CtorMemberPtrBad {
  int *p1;
  int *p2 [[ref_to_uninit]];
  CtorMemberPtrBad() : p1(&g_uninit), p2(&g_init) {}
  // expected-error@-1 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  // expected-error@-2 {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
};

struct CtorMemberPtrOK {
  int *p;
  int *q [[ref_to_uninit]];
  CtorMemberPtrOK() : p(&g_init), q(&g_uninit) {} // OK
};

struct CtorMemberRefBad {
  int &r;
  int &s [[ref_to_uninit]];
  CtorMemberRefBad() : r(g_uninit), s(g_init) {}
  // expected-error@-1 {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  // expected-error@-2 {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
};

struct CtorMemberRefOK {
  int &r;
  int &s [[ref_to_uninit]];
  CtorMemberRefOK() : r(g_init), s(g_uninit) {} // OK
};

// A member of an anonymous struct/union reaches the member-initializer site as
// an IndirectFieldDecl; the [[ref_to_uninit]] marking lives on the underlying
// field and is read from there.
struct CtorAnonStructMarkedOK {
  struct {
    int *p [[ref_to_uninit]];
  };
  CtorAnonStructMarkedOK() : p(&g_uninit) {} // OK: marked target, uninit source
};

struct CtorAnonStructMarkedBad {
  struct {
    int *p [[ref_to_uninit]];
  };
  CtorAnonStructMarkedBad() : p(&g_init) {} // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
};

struct CtorAnonStructUnmarkedBad {
  struct {
    int *p;
  };
  CtorAnonStructUnmarkedBad() : p(&g_uninit) {} // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};

struct CtorAnonUnionMarkedOK {
  union {
    int *p [[ref_to_uninit]];
    long *q;
  };
  CtorAnonUnionMarkedOK() : p(&g_uninit) {} // OK: marked target, uninit source
};

// A braced member-initializer is looked through to its single element, exactly
// like the variable-init site.
struct CtorBracedPtr {
  int *p;
  CtorBracedPtr() : p{&g_uninit} {} // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};

struct CtorBracedRef {
  int &r;
  CtorBracedRef() : r{g_uninit} {} // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};

// An out-of-line constructor definition is checked where it is defined; the
// enclosing constructor is still the CurContext there.
struct CtorOutOfLine {
  int *p;
  CtorOutOfLine();
};
CtorOutOfLine::CtorOutOfLine() : p(&g_uninit) {} // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}

// Cast and call sources reach the recognizer at the member-init site too: a
// cast of a [[ref_to_uninit]]-returning call propagates the marking.
struct CtorCastSource {
  int *p;
  CtorCastSource() : p((int *)alloc_void()) {} // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};

struct CtorSuppressWhole {
  int *p;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] CtorSuppressWhole() : p(&g_uninit) {} // OK: suppressed
};

struct CtorSuppressRule {
  int *p;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ref_to_uninit")]] CtorSuppressRule() : p(&g_uninit) {} // OK: suppressed
};

template <typename T>
struct CtorTmpl {
  int *p;
  CtorTmpl() : p(&g_uninit) {} // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};
template struct CtorTmpl<int>; // expected-note {{in instantiation of member function 'CtorTmpl<int>::CtorTmpl' requested here}}

// A never-instantiated class template stays silent: the deferred check never
// runs on the pattern.
template <typename T>
struct CtorTmplNever {
  int *p;
  CtorTmplNever() : p(&g_uninit) {}
};

// A written pointer member-initializer that binds to uninitialized memory
// yields exactly one ref_to_uninit error; the member is written, so
// ctor_uninit_member does not also fire.
struct NoDoubleFire {
  int *p;
  NoDoubleFire() : p(&g_uninit) {} // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
};

// A pointer member left uninitialized is not a ref_to_uninit binding (there is
// no written initializer) and yields only the ctor_uninit_member error.
struct UninitMemberOnly {
  int *p; // expected-note {{member 'p' declared here}}
  UninitMemberOnly() {} // expected-error {{constructor does not initialize member 'p' under profile 'std::init'}}
};

// std::init / ref_to_uninit (paper §5): a pointer/reference field initialized
// by an enclosing aggregate's init list is checked Decl-less, scoped to the
// field subobject, so the enclosing variable/argument/return is left to its own
// site (the variable site is not a pointer/reference here) and there is no
// double diagnostic.
struct AggPtr { int *p; };
struct AggPtrMarked { int *p [[ref_to_uninit]]; };
struct AggRef { int &r; };
struct AggRefMarked { int &r [[ref_to_uninit]]; };

void test_aggregate_pointer() {
  AggPtr a1{&g_uninit};       // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  AggPtr a2{&g_init};          // OK
  AggPtr a3 = {&g_uninit};     // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  AggPtrMarked m1{&g_init};    // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  AggPtrMarked m2{&g_uninit};  // OK
  (void)a1; (void)a2; (void)a3; (void)m1; (void)m2;
}

void test_aggregate_reference() {
  AggRef a1{g_uninit};         // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  AggRef a2{g_init};            // OK
  AggRefMarked m1{g_init};      // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  AggRefMarked m2{g_uninit};    // OK
  (void)a1; (void)a2; (void)m1; (void)m2;
}

void test_aggregate_designated() {
  AggPtr a{.p = &g_uninit}; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  AggPtr b{.p = &g_init};    // OK
  (void)a; (void)b;
}

struct AggNested { AggPtr inner; };

void test_aggregate_nested() {
  AggNested a{{&g_uninit}}; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  AggNested b{{&g_init}};    // OK
  (void)a; (void)b;
}

// An aggregate temporary built from an init list is checked the same way
// wherever it appears -- as a call argument, a return value, or a new-expression
// initializer. The enclosing pointer (the parameter, the return type, the
// new-expression result) is not a pointer/reference to the field's storage, so
// only the field binding is diagnosed.
void take_agg_ptr(AggPtr a);
AggPtr make_agg_bad() { return {&g_uninit}; } // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
AggPtr make_agg_ok() { return {&g_init}; }     // OK

void test_aggregate_temporary() {
  take_agg_ptr(AggPtr{&g_uninit});   // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  take_agg_ptr(AggPtr{&g_init});      // OK
  AggPtr *h = new AggPtr{&g_uninit}; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  delete h;
}

void test_aggregate_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] AggPtr a{&g_uninit};                        // OK: whole-profile suppress
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ref_to_uninit")]] AggPtr b{&g_uninit}; // OK: rule-targeted suppress
  (void)a; (void)b;
}

// Aggregate field init inside a template body is non-dependent here, so it
// fires at definition time and repeats when the local variable's
// initialization is rebuilt at instantiation; a never-instantiated template
// diagnoses at definition, once.
template <typename T>
void template_aggregate_bad() {
  AggPtr a{&g_uninit}; // expected-error 2 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)a;
}
template void template_aggregate_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_aggregate_bad<int>' requested here}}

template <typename T>
void template_aggregate_never() {
  AggPtr a{&g_uninit}; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)a;
}

// A thrown pointer copy-initializes the exception object, which cannot carry
// [[ref_to_uninit]], so it must not point to uninitialized memory. A read
// like `throw *p` is the read-through check's territory instead.
void throw_ptr_ok() { throw &g_init; } // OK
void throw_ptr_bad() { throw &g_uninit; } // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
void throw_marked_ptr_bad(int *p [[ref_to_uninit]]) { throw p; } // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
void throw_ptr_suppressed() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { throw &g_uninit; } // OK: suppressed
}

// A dependent thrown operand defers on the pattern and is rebuilt at
// instantiation, where the check fires per specialization. A fully
// non-dependent throw fires at definition time instead (see
// template_allglobal_bad): TreeTransform reuses it unchanged, so deferring
// would lose the diagnostic.
template <typename T>
void template_throw_bad() {
  throw (T *)&g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template void template_throw_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_throw_bad<int>' requested here}}

template <typename T>
void template_throw_nondependent_bad() {
  throw &g_uninit; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template void template_throw_nondependent_bad<int>();

// C++20 parenthesized aggregate initialization performs the same per-field
// bindings as the braced form and is checked identically.
void test_aggregate_paren() {
  AggPtr a1(&g_uninit);      // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  AggPtr a2(&g_init);        // OK
  AggPtrMarked m1(&g_init);  // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  AggPtrMarked m2(&g_uninit); // OK
  AggRef r1(g_uninit);       // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  AggRefMarked r2(g_init);   // expected-error {{reference marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  AggNested n1((AggPtr(&g_uninit))); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] AggPtr s(&g_uninit); // OK: suppressed
  (void)a1; (void)a2; (void)m1; (void)m2; (void)r1; (void)r2; (void)n1; (void)s;
}

template <typename T>
void template_aggregate_paren_bad() {
  AggPtr a(&g_uninit); // expected-error 2 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)a;
}
template void template_aggregate_paren_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_aggregate_paren_bad<int>' requested here}}

// A plain pointer *variable* with a braced initializer is checked once at its
// own variable site (EK_Variable); the aggregate field hooks are scoped to a
// member subobject, so this fires exactly once with no new duplicate.
void test_variable_braced_once() {
  int *p{&g_uninit}; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)p;
}

// std::init / ref_to_uninit (paper §5): a source whose syntactic form the
// recognizer does not model -- pointer arithmetic, an integer-to-pointer cast,
// or a call through a function pointer -- is unknown, not initialized. A marked
// target binds from such a source without error (it cannot be proven
// initialized, so rejecting it would be a false positive); an unmarked target
// also binds without error (a documented missed diagnostic). Neither run
// diagnoses these.
void test_unknown_pointer_arithmetic() {
  int *base [[ref_to_uninit]] = &g_uninit;
  int *p1 [[ref_to_uninit]] = base + 1; // OK
  int *p2 = base + 1;                    // OK
  (void)p1; (void)p2;
}

void test_unknown_int_to_ptr(long n) {
  int *p1 [[ref_to_uninit]] = reinterpret_cast<int *>(n); // OK
  int *p2 = reinterpret_cast<int *>(n);                    // OK
  (void)p1; (void)p2;
}

void test_unknown_fnptr_call(int *(*fp)()) {
  int *p1 [[ref_to_uninit]] = fp(); // OK
  int *p2 = fp();                    // OK
  (void)p1; (void)p2;
}

// The unknown classification propagates through the pass-through forms.
void test_unknown_passthrough(bool c) {
  int *base [[ref_to_uninit]] = &g_uninit;
  int *p1 [[ref_to_uninit]] = c ? base + 1 : &g_init; // OK: an unknown arm keeps the whole unknown
  int *p2 [[ref_to_uninit]] = (h(), base + 1);        // OK
  int *p3 [[ref_to_uninit]] = {base + 1};             // OK
  (void)p1; (void)p2; (void)p3;
}

// The unknown classification reaches the assignment, call-argument, and return
// sites unchanged.
void test_unknown_assignment(long n) {
  int *base [[ref_to_uninit]] = &g_uninit;
  int *p [[ref_to_uninit]] = &g_uninit;
  p = base + 1;                   // OK
  p = reinterpret_cast<int *>(n); // OK
  int *q = &g_init;
  q = base + 1;                   // OK
  (void)p; (void)q;
}

void test_unknown_call_argument(long n) {
  int *base [[ref_to_uninit]] = &g_uninit;
  take_uninit_ptr(base + 1);                   // OK
  take_uninit_ptr(reinterpret_cast<int *>(n)); // OK
  take_ptr(base + 1);                          // OK
}

[[ref_to_uninit]] int *ret_unknown_marked() {
  int *base [[ref_to_uninit]] = &g_uninit;
  return base + 1; // OK
}
int *ret_unknown_unmarked() {
  int *base [[ref_to_uninit]] = &g_uninit;
  return base + 1; // OK
}

// Regression guard: the fix narrows only the unknown case. A marked target
// bound from an affirmatively initialized source is still rejected, and an
// unmarked target from an affirmatively uninitialized source is still rejected.
void test_unknown_regression_guard() {
  int *m1 [[ref_to_uninit]] = &g_init;    // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *m2 [[ref_to_uninit]] = new int(5); // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  int *u1 = &g_uninit;                    // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)m1; (void)m2; (void)u1;
}

// A member call binds its implicit object parameter to the object argument
// (paper §7.2), and that parameter can never carry [[ref_to_uninit]], so a
// call on an object recognized as uninitialized storage is always the
// unmarked-direction violation. Every member-call flavor converts its object
// argument through the same funnel: dot and arrow calls, member operators,
// functor operator(), operator->, and conversion operators.
struct Callee {
  int m;
  int f() { return m; }
  static int sf() { return 0; }
  Callee &operator=(const Callee &);
  bool operator==(const Callee &) const;
  operator int() const;
  int *operator->();
  int operator()(int);
  int &operator[](int);
};

void test_member_call_on_uninit_object() {
  Callee s [[uninit]];
  s.f();     // expected-error {{calling member function 'f' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
  (&s)->f(); // expected-error {{calling member function 'f' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
}

void test_member_call_through_marked_pointer(Callee *p [[ref_to_uninit]]) {
  p->f();   // expected-error {{calling member function 'f' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
  (*p).f(); // expected-error {{calling member function 'f' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
}

// Member operators bind the same implicit object parameter, so whole-object
// assignment to an [[uninit]] class object -- previously unchecked through
// the overloaded operator= path -- is caught here too.
void test_member_operators_on_uninit_object() {
  Callee s [[uninit]];
  Callee t{};
  s = t;          // expected-error {{calling member function 'operator=' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
  (void)(s == t); // expected-error {{calling member function 'operator==' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
  int v = s;      // expected-error {{calling member function 'operator int' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
  s(1);           // expected-error {{calling member function 'operator()' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
  s[0] = 1;       // expected-error {{calling member function 'operator[]' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
  (void)v;
}

void test_member_arrow_on_uninit_object() {
  Callee s [[uninit]];
  (void)*(s.operator->()); // expected-error {{calling member function 'operator->' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
}

// A [[ref_to_uninit]]-returning reference function yields an uninitialized
// referent; calling a member function on it is the same violation.
[[ref_to_uninit]] Callee &get_uninit_callee();
void test_member_call_on_marked_call_result() {
  get_uninit_callee().f(); // expected-error {{calling member function 'f' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
}

Callee make_callee();
struct WithDtor {
  int m;
  void g();
  ~WithDtor();
};

// A static call operator has no implicit object parameter; the object
// argument is evaluated but its value never used, exactly like a static
// member function named through an object.
struct StaticCall {
  int m;
  static int operator()(int x) { return x; }
};

void test_member_call_silent_forms() {
  Callee s [[uninit]];
  s.sf();                 // OK: a static member uses no object argument
  (void)sizeof(s.f());    // OK: unevaluated
  using Unevaluated = decltype(s.f()); // OK: unevaluated
  (void)Unevaluated{};
  Callee t{};
  t.f();                  // OK: initialized object
  Callee{}.f();           // OK: a prvalue object is not uninitialized storage
  make_callee().f();      // OK: unmarked call result is trusted initialized
  WithDtor d [[uninit]];
  d.~WithDtor();          // OK: destruction is the deferred destroy_at slice
  StaticCall c [[uninit]];
  c(1);                   // OK: a static call operator uses no object argument
}

// A call through a pointer-to-member resolves no method at the call and
// bypasses the object-argument conversion -- the pointer-to-member analog of
// the call-through-function-pointer gap (a known gap, not an endorsement).
void test_member_call_through_pointer_to_member() {
  Callee s [[uninit]];
  int (Callee::*pmf)() = &Callee::f;
  (s.*pmf)(); // OK: known gap
}

// The object itself being unmarked keeps the trust decision: a class whose
// *member* is [[uninit]] may still have its member functions called (its
// constructor body may have assigned the member, paper §5.1/§5.2).
struct MemberOnlyUninit {
  int m [[uninit]];
  MemberOnlyUninit() { m = 1; }
  int get() { return m; }
};
void test_member_call_unmarked_object_trusted(MemberOnlyUninit &r) {
  MemberOnlyUninit o;
  o.get(); // OK
  r.get(); // OK: unknown-state reference parameter is not affirmatively uninit
}

// An explicit object member function initializes its object as an ordinary
// parameter, so the existing parameter binding check owns it (and its
// parameter *could* carry the marker).
struct ExplicitObj {
  int m;
  void f(this ExplicitObj &self);
};
void test_member_call_explicit_object() {
  ExplicitObj x [[uninit]];
  x.f(); // expected-error {{reference to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}

// A member with enable_if converts availability-check arguments under a
// SFINAE trap; the real call still diagnoses exactly once.
struct WithEnableIf {
  int m;
  void f() __attribute__((enable_if(true, "")));
};
void test_member_call_enable_if() {
  WithEnableIf s [[uninit]];
  s.f(); // expected-error {{calling member function 'f' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
}

void test_member_call_suppressed() {
  Callee s [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { s.f(); }         // OK: suppressed
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ref_to_uninit")]] { s.f(); } // OK
}

// A dependent object argument defers to instantiation, where the rebuilt call
// re-runs the funnel; a non-dependent call in a template fires at definition
// time and repeats when the call is rebuilt at instantiation (the local is
// remapped) -- the accepted repetition.
template <typename T>
void template_member_call_dependent_bad() {
  T s [[uninit]];
  s.f(); // expected-error {{calling member function 'f' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
}
template void template_member_call_dependent_bad<Callee>(); // expected-note {{in instantiation of function template specialization 'template_member_call_dependent_bad<Callee>' requested here}}

template <typename T>
void template_member_call_nondependent_bad() {
  Callee s [[uninit]];
  s.f(); // expected-error 2 {{calling member function 'f' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
}
template void template_member_call_nondependent_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_member_call_nondependent_bad<int>' requested here}}
