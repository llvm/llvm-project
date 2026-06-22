// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

struct Trivial { int x; };
struct AllInit { int x = 0; };
struct WithCtor { WithCtor(); int x; };
struct PartlyInit { int x; struct Inner { Inner(); } s; };
struct Nested { Trivial t; };
struct WithBase : Trivial {};
struct MarkedMember { int x [[uninit]]; };
struct MixedMarked { int a [[uninit]]; int b; };
struct NestedMarked { MarkedMember m; };

void test_aggregate() {
  Trivial a;                   // expected-error {{variable 'a' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  Trivial b = {1};
  Trivial c = {};
  Trivial d{};
  Trivial e [[uninit]];
  (void)a; (void)b; (void)c; (void)d; (void)e;
}

void test_nested_and_base() {
  PartlyInit a;                // expected-error {{variable 'a' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  Nested b;                    // expected-error {{variable 'b' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  WithBase c;                  // expected-error {{variable 'c' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)a; (void)b; (void)c;
}

void test_trusted() {
  // A type with a user-provided default constructor is trusted (the
  // constructor is checked separately), and a non-static data member
  // initializer covers the member.
  WithCtor a;
  AllInit b;
  (void)a; (void)b;
}

void test_marked_members() {
  // A type whose only indeterminate scalars are [[uninit]] is trusted;
  // those members are acknowledged uninitialized (paper §6.2), even through a
  // nesting level. A mixed type still fires for its unmarked scalar.
  MarkedMember a;
  NestedMarked b;
  MixedMarked c; // expected-error {{variable 'c' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)a; (void)b; (void)c;
}

void test_arrays() {
  // An automatic array of a class type whose default-init leaves a scalar
  // subobject indeterminate is diagnosed via the base element type (paper
  // section 6).
  Trivial a[3];                // expected-error {{variable 'a' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  Trivial b[2][3];             // expected-error {{variable 'b' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  int c[5];                    // expected-error {{variable 'c' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  Trivial d[2] = {};
  Trivial e[2] = {{1}, {2}};
  [[uninit]] Trivial f[3];
  WithCtor g[3];
  AllInit h[3];
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
}

void test_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_decl")]] Trivial a;
  (void)a;
}

// Non-local aggregates are zero-initialized at static-init time, so they are
// not diagnosed.
Trivial g_trivial;
