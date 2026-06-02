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
