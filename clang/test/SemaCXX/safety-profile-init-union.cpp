// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

union U { int x; float y; };

U g_union [[uninit]]; // expected-error {{'[[uninit]]' cannot be applied to a variable of union type under profile 'std::init'}}

void test_union_var() {
  U a [[uninit]]; // expected-error {{'[[uninit]]' cannot be applied to a variable of union type under profile 'std::init'}}
  (void)a;
}

void test_union_var_suppressed() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] U a [[uninit]];
  (void)a;
}

union MarkedMember {
  int x [[uninit]]; // expected-error {{'[[uninit]]' cannot be applied to a union member under profile 'std::init'}}
  float y;
};

// The marker checks key on the base element type: an array of unions is
// banned exactly like a single union object (paper section 5.6).
[[uninit]] U g_union_arr[2]; // expected-error {{'[[uninit]]' cannot be applied to a variable of union type under profile 'std::init'}}

void test_union_array_var() {
  [[uninit]] U a[2];    // expected-error {{'[[uninit]]' cannot be applied to a variable of union type under profile 'std::init'}}
  [[uninit]] U b[2][3]; // expected-error {{'[[uninit]]' cannot be applied to a variable of union type under profile 'std::init'}}
  (void)a; (void)b;
}

// A union-typed data member of a non-union class cannot carry the marker
// either: delayed initialization by assigning one of its members would be
// just as erroneous there (paper section 5.6).
struct HasMarkedUnionMember {
  U u [[uninit]];      // expected-error {{'[[uninit]]' cannot be applied to a data member of union type under profile 'std::init'}}
  [[uninit]] U arr[2]; // expected-error {{'[[uninit]]' cannot be applied to a data member of union type under profile 'std::init'}}
};

// A marker on a union member of a non-enforced profile is silently accepted;
// exercised by the no-profiles run above.

// A non-union class member may carry the marker (it is not banned here).
struct NotUnion {
  int x [[uninit]];
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

// A union whose only members are unnamed bit-fields has nothing that
// default-initialization could leave indeterminate -- unnamed bit-fields are
// not members and no in-language initializer exists for them -- but an
// unnamed bit-field alongside a real member changes nothing.
union BitFieldOnly { int : 4; };
union BitFieldPlusMember { int : 4; int i; };
void test_bitfield_only_union() {
  BitFieldOnly a;       // OK: nothing to initialize
  BitFieldPlusMember b; // expected-error {{variable 'b' of union type must be initialized under profile 'std::init'}}
  (void)a; (void)b;
}

// A union data member that a constructor leaves uninitialized is diagnosed; one
// initialized via its member-initializer is accepted.
struct HasUnionMember {
  U u;                                // expected-note {{member 'u' declared here}}
  int z;
  HasUnionMember() : z(0) {}          // expected-error {{constructor does not initialize member 'u' under profile 'std::init'}}
  HasUnionMember(int) : u{1}, z(0) {}
};

// A dependent local that substitutes to a union type is deferred on the pattern
// and fires union_marker at instantiation, not on the template.
template <typename T>
void template_union_marker() {
  T x [[uninit]]; // #template-union-marker
  (void)x;
}
template void template_union_marker<U>(); // expected-note {{in instantiation of function template specialization 'template_union_marker<U>' requested here}}
// expected-error@#template-union-marker {{'[[uninit]]' cannot be applied to a variable of union type under profile 'std::init'}}

// A dependent local that substitutes to an *array of* unions fires the same
// way (base element type) at instantiation.
template void template_union_marker<U[2]>(); // expected-note {{in instantiation of function template specialization 'template_union_marker<U[2]>' requested here}}
// expected-error@#template-union-marker {{'[[uninit]]' cannot be applied to a variable of union type under profile 'std::init'}}

// An uninstantiated pattern never reaches phase 7, so the marker is silent.
template <typename T>
void template_union_never_instantiated() {
  T x [[uninit]];
  (void)x;
}
