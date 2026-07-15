// RUN: %clang_cc1 -fsyntax-only -verify -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// std::init / uninit_write (paper §5.4-§5.6): a scalar store to a proper
// subobject of a named [[uninit]] entity is banned delayed initialization --
// only writing the whole named entity initializes it (paper §4.5), and only
// whole-object construct_at could make a piecemeal-initialized object good.

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

namespace std { enum class byte : unsigned char {}; }

struct Pair { int x; int y; };

bool cond();

// A store to the whole named entity is its initialization (paper §4.5), for
// the marked local itself and for a directly-targeted marked member.
void test_whole_entity_store_is_initialization() {
  int x [[uninit]];
  x = 7; // OK: the write initializes x
  (void)x;
}

struct WithMarkedMember { int m [[uninit]]; };
void test_direct_marked_member_store(WithMarkedMember &o) {
  o.m = 1; // OK: the store targets the marked member itself
}

// The §5.2 constructor-body pattern is untouched: a current-object member
// store reaches the member through `this` (a pointer), in every spelling.
struct CtorBody {
  int m [[uninit]];
  CtorBody() {
    m = 1;         // OK
    this->m = 2;   // OK
    (*this).m = 3; // OK
  }
};

// Member stores below a named [[uninit]] object (paper §5.4).
void test_member_store(bool c) {
  Pair s [[uninit]];
  s.x = 1;     // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  (&s)->x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  Pair t [[uninit]];
  (c ? s : t).x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}

  WithMarkedMember b [[uninit]];
  b.m = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}

  Pair u; // expected-error {{variable 'u' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  u.x = 1; // OK: 'u' is not marked; its diagnostic is uninit_decl's, at the declaration
}

// A marked *member* below the top level bans the store the same way, on a
// named object or on the current object -- a class-type member has no legal
// piecemeal path (only whole-object construct_at, which is unmodeled).
struct HasAgg { Pair agg [[uninit]]; };
void test_nested_marked_member_store(HasAgg &h) {
  h.agg.x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}
struct SelfAgg {
  Pair agg [[uninit]];
  void set() {
    agg.x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  }
};

// Element stores of [[uninit]] arrays are the §5.5 random-access-init ban.
void test_element_store(int i) {
  [[uninit]] int a[2];
  a[0] = 1; // expected-error {{writing an element of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  a[i] = 1; // expected-error {{writing an element of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  *a = 1;   // expected-error {{writing an element of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  [[uninit]] int m2[2][2];
  m2[1][0] = 1; // expected-error {{writing an element of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}

// A marked array *member*'s elements are proper subobjects of the marked
// array, so element stores are banned even on the current object.
struct WithArrMember {
  [[uninit]] int a[2];
  void set() {
    a[0] = 1; // expected-error {{writing an element of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  }
};
void test_member_array_element_store(WithArrMember &w) {
  w.a[0] = 1; // expected-error {{writing an element of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}

// Member assignment to a marker-retaining union is the §5.6 ban (the marker
// itself is union_marker's; here it is suppressed and retained).
union U { int x; float y; };
void test_union_member_store() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] U u [[uninit]];
  u.x = 9; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}

// Compound assignments and built-in increments store like plain assignment,
// and every one of them also reads the old value: the shift forms load
// through their LHS promotion (DefaultLvalueConversion), the rest are checked
// at the operator sites, so each fires the read-through diagnostic exactly
// once alongside the write.
void test_compound_and_incdec_stores() {
  Pair s [[uninit]];
  s.x += 1;  // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}} \
             // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  s.x <<= 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}} \
             // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  ++s.x;     // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}} \
             // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  s.x--;     // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}} \
             // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  [[uninit]] int a[2];
  --a[0];    // expected-error {{writing an element of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}} \
             // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
}

// Writes through a [[ref_to_uninit]] pointer or reference: for a built-in
// type the write is the pointee's initialization (paper §4.5), so they are
// legal -- and a whole-`*p` store credits the pointee as initialized in
// parse order, so the compound assignment below may read the value it wrote.
// An element store (p[3]) neither credits nor invalidates (§5.4/§5.5). A
// compound assignment through a still-uncredited marker reads uninitialized
// memory (full coverage in safety-profile-init-ref-to-uninit.cpp).
[[ref_to_uninit]] int &get_uninit_ref();
void test_write_through_ref_to_uninit(int *p [[ref_to_uninit]],
                                      int &r [[ref_to_uninit]],
                                      Pair *ptr [[ref_to_uninit]]) {
  *p = 5;              // OK (and credits p's pointee)
  p[3] = 0;            // OK (no credit, no invalidation)
  *p += 1;             // OK: the whole-*p store above credited the pointee
  r = 5;               // OK
  ptr->x = 5;          // OK
  get_uninit_ref() = 5; // OK
}

// std::byte may be left uninitialized and manipulated freely (paper §4.5).
void test_byte_exempt() {
  [[uninit]] std::byte b[2];
  b[0] = std::byte{1}; // OK
}

// The §5.4 sanctioned route for whole-object (re)initialization: a
// [[ref_to_uninit]]-taking construct_at, whose parameter binding accepts &s.
template <class T, class... Args>
T *construct_at(T *p [[ref_to_uninit]], Args &&...args);

void test_construct_at_pattern() {
  Pair s [[uninit]];
  construct_at(&s, 1, 2); // OK: the marked parameter accepts &s
  // The address escape earns no store credit -- paper §6.2 reserves
  // callee-initialization for now_init(); only whole-entity stores credit
  // (stores-only policy) -- so the subobject write below stays an error.
  s.x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}

// Suppression: whole-profile and rule-targeted, on statements and on the
// enclosing declaration; a different rule does not cover the store.
void test_suppress_stmt() {
  Pair s [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { s.x = 1; } // OK
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_write")]] { s.y = 2; } // OK
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] {
    s.x = 3; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  }
}

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "uninit_write")]]
void test_suppress_decl() {
  Pair s [[uninit]];
  s.x = 1; // OK: the function-level suppression covers the body
}

// Like every Decl-less expression check, the store check fires at definition
// time when its target is non-dependent, and repeats when the local s is
// remapped and the assignment rebuilt at instantiation -- the accepted
// repetition. A dependent target (template_write_dependent_bad) defers on the
// pattern and fires once per violating specialization.
template <typename T>
void template_write_bad() {
  Pair s [[uninit]];
  s.x = 1; // expected-error 2 {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}
template void template_write_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_write_bad<int>' requested here}}

template <typename T>
void template_write_dependent_bad() {
  T s [[uninit]];
  s.x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}
template void template_write_dependent_bad<Pair>(); // expected-note {{in instantiation of function template specialization 'template_write_dependent_bad<Pair>' requested here}}

// A never-instantiated pattern diagnoses its non-dependent store at
// definition time, exactly once.
template <typename T>
void template_write_never_instantiated() {
  Pair s [[uninit]];
  s.x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}

// A literal-false if-constexpr condition makes the then-branch a discarded
// statement context already at pattern parse, so its store stays silent; the
// live else-branch fires at definition time and repeats when the branch is
// rebuilt at instantiation.
template <typename T>
void template_write_discarded_branch() {
  Pair s [[uninit]];
  if constexpr (false) {
    s.x = 1;
  } else {
    s.x = 2; // expected-error 2 {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  }
}
template void template_write_discarded_branch<int>(); // expected-note {{in instantiation of function template specialization 'template_write_discarded_branch<int>' requested here}}

// An all-global store is *reused* by TreeTransform at instantiation (its
// Build* never re-runs), so deferring would silently lose the diagnostic; it
// is checked at definition time -- exactly one error, with no instantiation
// note, whether or not the template is ever instantiated.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "static_marker")]] [[uninit]] Pair g_uninit_pair;

template <typename T>
void template_allglobal_write_bad() {
  g_uninit_pair.x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}
template void template_allglobal_write_bad<int>();

template <typename T>
void template_allglobal_write_never_instantiated() {
  g_uninit_pair.x = 1; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
}

// A store that violates two rules at one location -- the subobject write into
// the [[uninit]] object and the unmarked-pointer binding of its member --
// fires both at definition time, and both repeat on the instantiation
// rebuild.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init, rule: "static_marker")]] [[uninit]] int g_uninit_int;
struct WithPtrMember { int x; int *p; };

template <typename T>
void template_two_rules_bad() {
  WithPtrMember s [[uninit]];
  s.p = &g_uninit_int; // expected-error 2 {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}} \
                       // expected-error 2 {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}
template void template_two_rules_bad<int>(); // expected-note {{in instantiation of function template specialization 'template_two_rules_bad<int>' requested here}}

template <typename T>
void template_two_rules_never_instantiated() {
  WithPtrMember s [[uninit]];
  s.p = &g_uninit_int; // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}} \
                       // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
}

// Parse-order whole-entity store credit (paper §4.2/§4.5): assigning the
// whole [[uninit]] entity is its initialization, so bindings after the store
// (in parse order) treat it as initialized -- and the paper's reverse
// direction applies: the credited entity now REQUIRES an unmarked target
// (§4.2's `p4 = &x` error). Purely parse-order, no flow analysis: a store
// under a condition credits everything after it (§1.2 "consider all branches
// executed" -- the untaken path is a missed diagnostic, never a false
// positive).
void take_int_ptr(int *q);
void test_whole_store_credit() {
  int u [[uninit]];
  u = 5;
  int *q = &u;      // OK: u is initialized (rejected before the store)
  take_int_ptr(&u); // OK
  int &br = u;      // OK
  int *r [[ref_to_uninit]] = &u; // expected-error {{pointer marked '[[ref_to_uninit]]' must refer to uninitialized memory under profile 'std::init'}}
  (void)q; (void)br; (void)r;
}

void test_binding_before_store() {
  int u [[uninit]];
  int *q = &u; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  u = 5;
  (void)q;
}

void test_conditional_store_credit(bool c) {
  int u [[uninit]];
  if (c)
    u = 5;
  int *q = &u; // OK: parse-order credit (§1.2); the untaken path is an
               // accepted missed diagnostic
  (void)q;
}

// Compound assignment and ++/-- store, so they credit the whole entity --
// while their own old-value read keeps the flow-based read-before-init
// error (recorded after the pre-store checks: no self-crediting).
void test_compound_store_credit() {
  int u [[uninit]]; // expected-note {{variable 'u' is declared here}}
  u += 1;       // expected-error {{variable 'u' is read before initialization under profile 'std::init'}}
  int *q = &u;  // OK: the compound store credited u
  int v [[uninit]]; // expected-note {{variable 'v' is declared here}}
  ++v;          // expected-error {{variable 'v' is read before initialization under profile 'std::init'}}
  int *qv = &v; // OK
  int w [[uninit]]; // expected-note {{variable 'w' is declared here}}
  w = w + 1;    // expected-error {{variable 'w' is read before initialization under profile 'std::init'}}
  (void)q; (void)qv;
}

// Class-typed whole-object assignment never credits: it resolves to a member
// operator= -- already rejected as a call on uninitialized storage -- and
// never reaches the built-in assignment funnel (crediting an erroneous
// statement would misstate the object's state). The class remedy remains
// construct_at through [[ref_to_uninit]] (paper §4.5; unmodeled slice).
void test_class_store_never_credits() {
  Pair s [[uninit]];
  s = Pair{1, 2}; // expected-error {{calling member function 'operator=' binds its implicit object parameter to uninitialized memory under profile 'std::init'}}
  s.x = 3;        // expected-error {{writing a member of an '[[uninit]]' object does not initialize it under profile 'std::init'; initialize the whole object}}
  int y = s.x;    // expected-error {{read of a subobject of an '[[uninit]]' object accesses uninitialized memory under profile 'std::init'}}
  (void)y;
}

// A store in an unevaluated or discarded context never executes, so it earns
// no credit.
void test_no_credit_contexts() {
  int u [[uninit]];
  (void)sizeof(u = 5); // expected-warning {{expression with side effects has no effect in an unevaluated context}} \
                       // no-profiles-warning {{expression with side effects has no effect in an unevaluated context}}
  int *q = &u; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int v [[uninit]];
  using TT = decltype((v = 5)); // expected-warning {{expression with side effects has no effect in an unevaluated context}} \
                                // no-profiles-warning {{expression with side effects has no effect in an unevaluated context}}
  int *qv = &v; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  int w [[uninit]];
  if constexpr (false) { w = 5; }
  int *qw = &w; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
  (void)q; (void)qv; (void)qw;
}

// A suppressed store still initializes: the credit is recorded regardless
// of [[profiles::suppress]], so suppression cannot manufacture later false
// positives.
void test_suppressed_store_credits() {
  int u [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] { u = 5; }
  int *q = &u; // OK: the suppressed store still credited u
  (void)q;
}

// A nested assignment credits both targets: the inner assignment completes
// (and records) before the outer one.
void test_nested_assignment_credit() {
  int u [[uninit]];
  int v [[uninit]];
  u = (v = 5);
  int *qu = &u; // OK
  int *qv = &v; // OK
  (void)qu; (void)qv;
}

// The credit keys on the unique VarDecl: a same-named sibling-scope local is
// a distinct declaration, so credit does not leak between them.
void test_sibling_scope_credit(bool c) {
  if (c) {
    int u [[uninit]];
    u = 5;
    int *q = &u; // OK: this u is credited
    (void)q;
  } else {
    int u [[uninit]];
    int *q = &u; // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
    (void)q;
  }
}
