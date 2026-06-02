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

void test_aggregate() {
  Trivial a;                   // expected-error {{variable 'a' must be initialized or marked '[[uninitialized]]' under profile 'std::init'}}
  Trivial b = {1};
  Trivial c = {};
  Trivial d{};
  Trivial e [[uninitialized]];
  (void)a; (void)b; (void)c; (void)d; (void)e;
}

void test_nested_and_base() {
  PartlyInit a;                // expected-error {{variable 'a' must be initialized or marked '[[uninitialized]]' under profile 'std::init'}}
  Nested b;                    // expected-error {{variable 'b' must be initialized or marked '[[uninitialized]]' under profile 'std::init'}}
  WithBase c;                  // expected-error {{variable 'c' must be initialized or marked '[[uninitialized]]' under profile 'std::init'}}
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

void test_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_decl")]] Trivial a;
  (void)a;
}

// Non-local aggregates are zero-initialized at static-init time, so they are
// not diagnosed.
Trivial g_trivial;
