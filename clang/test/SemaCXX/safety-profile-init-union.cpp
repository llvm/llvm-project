// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

union U { int x; float y; };

U g_union [[uninitialized]]; // expected-error {{'[[uninitialized]]' cannot be applied to a variable of union type under profile 'std::init'}}

void test_union_var() {
  U a [[uninitialized]]; // expected-error {{'[[uninitialized]]' cannot be applied to a variable of union type under profile 'std::init'}}
  (void)a;
}

void test_union_var_suppressed() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] U a [[uninitialized]];
  (void)a;
}

union MarkedMember {
  int x [[uninitialized]]; // expected-error {{'[[uninitialized]]' cannot be applied to a union member under profile 'std::init'}}
  float y;
};

// A marker on a union member of a non-enforced profile is silently accepted;
// exercised by the no-profiles run above.

// A non-union class member may carry the marker (it is not banned here).
struct NotUnion {
  int x [[uninitialized]];
};

union WithNSDMI { int x = 0; float y; };
// Defining a union constructor must not fire ctor_uninit_member for the other
// members (they are mutually exclusive).
union WithUserCtor { int x; float y; WithUserCtor() : x(0) {} };

void test_uninit_union_object() {
  U a;            // expected-error {{variable 'a' of union type must be initialized under profile 'std::init'}}
  U b = {1};
  U c{};
  WithNSDMI d;    // OK: a default member initializer initializes a member
  WithUserCtor e; // OK: a user-provided default constructor is trusted
  (void)a; (void)b; (void)c; (void)d; (void)e;
}

void test_uninit_union_suppressed() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] U a;
  (void)a;
}

// A union data member that a constructor leaves uninitialized is diagnosed; one
// initialized via its member-initializer is accepted.
struct HasUnionMember {
  U u;                                // expected-note {{member 'u' declared here}}
  int z;
  HasUnionMember() : z(0) {}          // expected-error {{constructor does not initialize member 'u' under profile 'std::init'}}
  HasUnionMember(int) : u{1}, z(0) {}
};
