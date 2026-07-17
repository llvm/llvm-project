// All violations share one TU with a leading unrelated error: the early error
// disables the analysis-based-warnings pass for later functions, so this also
// verifies that the local-aggregate member check keeps diagnosing through the
// post-error rerun.
// RUN: %clang_cc1 -fsyntax-only -verify=expected,common -fprofiles -std=c++23 -Wno-uninitialized %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles,common -std=c++23 -Wno-uninitialized %s

// std::init: an [[uninit]] scalar member of a constructor-less aggregate
// local (the paper §5.3 "class exposing uninitialized members" pattern) is
// given a value by a plain member store; a read before the member is
// definitely assigned on every path is diagnosed by a per-function
// definite-assignment pass, the local-variable analog of the ctor-body check.

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

namespace std { enum class byte : unsigned char {}; }

int leading_unrelated_error = undeclared_identifier;
// common-error@-1 {{use of undeclared identifier 'undeclared_identifier'}}

struct Agg {
  int m [[uninit]]; // expected-note 18 {{member 'm' declared here}}
};
void take_ref(Agg &);
// The pointee is uninitialized memory, so the parameter carries the marker
// (the unmarked spelling is the ref_to_uninit binding rule's to reject).
void take_ptr(int *p [[ref_to_uninit]]);

int test_read_before_any_write() {
  Agg a;
  return a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

int test_read_then_write() {
  Agg a;
  int v = a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
  a.m = 1;
  return v;
}

int test_branch_one_path(bool c) {
  Agg a;
  if (c)
    a.m = 1;
  return a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// A compound assignment and a built-in ++/-- read the old value before
// writing it.
void test_compound_reads_old_value() {
  Agg a;
  a.m += 1; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

void test_incdec_reads_old_value() {
  Agg a;
  a.m++; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// A loop body may run zero times, so an assignment inside it does not reach a
// read after the loop...
int test_loop_may_not_run(int n) {
  Agg a;
  for (int i = 0; i < n; ++i)
    a.m = i;
  return a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// ...and a read at the top of the body precedes the first iteration's write.
int test_loop_read_first_iteration(int n) {
  Agg a;
  int t = 0;
  for (int i = 0; i < n; ++i) {
    t += a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    a.m = i;
  }
  return t;
}

// sizeof neither reads the member (unevaluated) nor escapes the object.
int test_sizeof_neither_reads_nor_escapes() {
  Agg a;
  (void)sizeof(a.m);
  return a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// An [[uninit]] member inherited from a constructor-less non-virtual base is
// tracked like the class's own (nothing can have assigned it earlier).
struct Base {
  int bm [[uninit]]; // expected-note {{member 'bm' declared here}}
};
struct Derived : Base {
  int dm = 0;
};
int test_base_subtree_member() {
  Derived d;
  return d.bm; // expected-error {{member 'bm' is read before initialization under profile 'std::init'}}
}

int test_write_then_read() {
  Agg a;
  a.m = 5;
  return a.m; // OK
}

int test_branch_both_paths(bool c) {
  Agg a;
  if (c)
    a.m = 1;
  else
    a.m = 2;
  return a.m; // OK
}

// Any appearance of the variable outside a recognized member read or write
// conservatively marks every member assigned: the address may be used to
// initialize the object (construct_at, memcpy, an initializing callee).
// These pin the interim (pre-now_init()) leniency, paper §6.2; contrast the
// ctor-body pass's strict assignment-only crediting
// (safety-profile-init-ctor-body.cpp, Escape*).
int test_escape_address_of_object() {
  Agg a;
  (void)&a;
  return a.m; // OK: escaped
}

int test_escape_address_of_member() {
  Agg a;
  take_ptr(&a.m);
  return a.m; // OK: escaped
}

int test_escape_reference_binding() {
  Agg a;
  take_ref(a);
  return a.m; // OK: escaped
}

int test_escape_lambda_capture() {
  Agg a;
  auto init = [&] { a.m = 5; };
  init();
  return a.m; // OK: the capture escapes the object
}

// Placement new takes the object's address -- the same escape as &a.
void *operator new(__SIZE_TYPE__, void *p) noexcept;
int test_escape_placement_new() {
  Agg a;
  new (&a) Agg{1};
  return a.m; // OK: escaped
}

// A class with a user-provided constructor is trusted (paper §5.1): its
// constructor body may have assigned the member, which local analysis cannot
// see. This pins the deliberate trust decision for non-current-object member
// reads.
struct Slot {
  int y [[uninit]];
  Slot() {}
};
int test_user_provided_ctor_trusted() {
  Slot uu;
  return uu.y; // OK: trusted
}

// A base with a user-provided constructor keeps its members untracked, even
// under a constructor-less derived class.
struct TrustedBase {
  int tm [[uninit]];
  TrustedBase() {}
};
struct DerivedFromTrusted : TrustedBase {};
int test_trusted_base_member() {
  DerivedFromTrusted d;
  return d.tm; // OK: trusted
}

// A value-initializing written form gives every member a value; only the
// bare `Agg a;` form (the implicit no-op default-construction) is tracked.
int test_value_initialized_forms() {
  Agg a{};
  Agg b = {};
  Agg c = Agg();
  return a.m + b.m + c.m; // OK
}

// A copy does NOT give the [[uninit]] member a value -- it copies
// indeterminate bits (a copy does not inherit initialization, paper §5.2)
// -- but the source's per-member state is unknowable for an untracked
// source, so such copies stay untracked: a known gap, never a false
// positive.
Agg make_agg();
int test_copy_from_untracked_source() {
  Agg e = make_agg();
  return e.m; // OK: known gap (untracked source)
}

// A by-value parameter is a copy of the caller's argument, and a copy does
// not inherit initialization (§5.2): its marked members are tracked from an
// unassigned start. This is the call-boundary twin of the ctor-body pass's
// deliberate strictness -- the paper hands uninitialized-capable storage
// across calls via marked pointers/references (§4.3), not by-value slots --
// so a caller-initialized member is rejected all the same; escapes and
// [[profiles::suppress]] are the remedies.
int test_byvalue_parameter_tracked(Agg p) {
  return p.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

int test_byvalue_parameter_assigned(Agg p) {
  p.m = 5;
  return p.m; // OK
}

int test_byvalue_parameter_escape(Agg p) {
  take_ref(p);
  return p.m; // OK: escaped
}

// Re-passing the parameter by value is an escape like any other bare use
// (the copy-constructor argument reference is not a tracked-copy DeclStmt).
void use_agg(Agg);
int test_byvalue_parameter_repassed(Agg p) {
  use_agg(p);
  return p.m; // OK: escaped
}

// A copy from a by-value parameter chains the tracking: the parameter
// starts unassigned, so the copy does too.
int test_copy_from_parameter(Agg other) {
  Agg d = other;
  return d.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// A reference parameter aliases the caller's own object -- not a copy --
// and stays untracked.
int test_reference_parameter_untracked(Agg &r) {
  return r.m; // OK
}

// A local that is itself [[uninit]]-marked is the parse-time rules'
// territory: the read-through check owns its subobject reads, and exactly one
// diagnostic fires.
int test_marked_local_owned_by_read_through() {
  Agg s [[uninit]];
  return s.m; // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
}

// A static local is zero-initialized, never tracked.
int test_static_local_untracked() {
  static Agg a;
  return a.m; // OK
}

// A member of an anonymous struct is reached through an IndirectFieldDecl
// chain, not a direct `a.m` access, so it is not tracked -- consistent with
// the anonymous-aggregate skips in the ctor-body pass and R5 (a known gap).
struct HasAnon {
  struct {
    int m [[uninit]];
  };
};
int test_anonymous_member_untracked() {
  HasAnon a;
  return a.m; // OK: known gap
}

// An array of aggregates is not tracked (element tracking is the deferred
// construct_at slice).
int test_array_of_aggregates_untracked() {
  Agg arr[2];
  return arr[0].m; // OK: known gap
}

// A union local is never tracked: the harvest rejects union types (their
// members are mutually exclusive, and [[uninit]] on a union member is banned
// by union_marker anyway -- suppressed on the function to build the fixture).
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "union_marker")]]
int test_union_local_untracked() {
  union U {
    int x [[uninit]];
  };
  U u = {1};
  return u.x; // OK
}

// std::byte members are exempt (paper §4.5), so a byte-only aggregate has
// nothing to track.
struct ByteBox {
  std::byte b [[uninit]];
};
std::byte test_byte_member_exempt() {
  ByteBox x;
  return x.b; // OK
}

void test_suppress_stmt() {
  Agg a;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] {
    int v = a.m; // OK: suppressed
    (void)v;
  }
}

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init)]]
int test_suppress_decl() {
  Agg a;
  return a.m; // OK: suppressed
}

// A lambda body's own locals are tracked when the lambda's call operator is
// analyzed.
void test_lambda_own_local() {
  auto f = [] {
    Agg a;
    return a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
  };
  (void)f;
}

// A template function's body is analyzed per instantiation.
template <typename T>
int template_local_member_read() {
  Agg a;
  return a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}
template int template_local_member_read<int>(); // expected-note {{in instantiation of function template specialization 'template_local_member_read<int>' requested here}}

// ============================================================
// Copies of tracked locals
// ============================================================

// A copy of a tracked local inherits the source's per-member state at the
// copy point -- a copy does not inherit initialization (paper §5.2), it
// inherits whatever state the source has -- so reading the copy's member is
// exactly as (in)valid as reading the source's was there.
int test_copy_read_before_source_assigned() {
  Agg a;
  Agg b = a;
  return b.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

int test_copy_after_source_assigned() {
  Agg a;
  a.m = 5;
  Agg b = a;
  return b.m; // OK: the source was assigned at the copy point
}

int test_copy_then_dest_assigned() {
  Agg a;
  Agg b = a;
  b.m = 1;
  return b.m; // OK
}

// The copy consumes the source ref without escaping it: the source keeps
// its own (unassigned) state.
int test_copy_keeps_source_tracked() {
  Agg a;
  Agg b = a;
  b.m = 1;
  return a.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// State transfers at the copy point, not later: a source assignment after
// the copy does not reach the copy.
int test_copy_point_state() {
  Agg a;
  Agg b = a;
  a.m = 5;
  return b.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// The all-branches rule (§1.2) applies through the copy.
int test_copy_after_branch(bool c) {
  Agg a;
  if (c)
    a.m = 5;
  Agg b = a;
  return b.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// State flows through a chain of copies (harvested to a fixpoint, so the
// chain resolves regardless of declaration order in the CFG's block list).
int test_copy_of_copy() {
  Agg a;
  a.m = 5;
  Agg b = a;
  Agg c = b;
  return c.m; // OK
}

int test_copy_of_copy_unassigned() {
  Agg a;
  Agg b = a;
  Agg c = b;
  return c.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// Move construction transfers state the same way (for these classes a move
// is a copy; the explicit-cast peel resolves the directly named source).
int test_move_construction() {
  Agg a;
  Agg b = static_cast<Agg &&>(a);
  return b.m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
}

// Paren and brace copy forms behave identically to the `=` form.
int test_copy_forms() {
  Agg a;
  a.m = 5;
  Agg b(a);
  Agg c{a};
  return b.m + c.m; // OK
}

// An escape of the copy credits the copy, like any tracked local.
int test_copy_escape() {
  Agg a;
  Agg b = a;
  take_ref(b);
  return b.m; // OK: escaped
}
