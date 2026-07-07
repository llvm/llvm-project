// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

struct PlainField {
  int m [[uninit]];
};

struct PlainFieldPrefix {
  [[uninit]] int m;
};

struct FieldWithNSDMI {
  int m [[uninit]] = 0; // expected-error {{variable 'm' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
};

struct FieldWithNSDMIPrefix {
  [[uninit]] int m = 0; // expected-error {{variable 'm' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
};

// A static data member is a zero-initialized static object, so its definition
// is rejected by static_marker (paper section 4.2), even though the marker
// attaches to the declaration; the error fires at the out-of-line definition.
struct WithStaticDataMember {
  static int s [[uninit]];
  [[uninit]] static int t;
};
int WithStaticDataMember::s; // expected-error {{'[[uninit]]' cannot be applied to variable 's' with static storage duration under profile 'std::init'; it is zero-initialized}}
int WithStaticDataMember::t; // expected-error {{'[[uninit]]' cannot be applied to variable 't' with static storage duration under profile 'std::init'; it is zero-initialized}}

struct MultipleFields {
  int a [[uninit]];
  int b = 0;
  int c [[uninit]] = 0; // expected-error {{variable 'c' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
};

template <typename T>
struct DependentField {
  T m [[uninit]]; // #dependent-field-member
};
template struct DependentField<int>; // OK: a non-pointer, non-union member

// The marker is deferred on the dependent pattern and re-checked once the
// substituted type is known, so a pointer member fires pointer_marker at
// instantiation (paper section 4.1).
template struct DependentField<int *>; // expected-note {{in instantiation of template class 'DependentField<int *>' requested here}}
// expected-error@#dependent-field-member {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}

// The deferral is keyed on the member being templated, not on its type being
// dependent: a literally non-dependent pointer member inside a template still
// defers on the pattern and fires once, at instantiation.
template <typename T>
struct NonDependentPtrField {
  int *m [[uninit]]; // #nondependent-ptr-field
};
template struct NonDependentPtrField<int>; // expected-note {{in instantiation of template class 'NonDependentPtrField<int>' requested here}}
// expected-error@#nondependent-ptr-field {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}

// An uninstantiated pattern is not yet a phase-7 entity, so nothing fires.
template <typename T>
struct DependentFieldNeverInstantiated {
  T m [[uninit]];
};

// A dependent member instantiating to a *reference* type is rejected at
// instantiation and the marker dropped, mirroring the parse-time rejection of
// a non-dependent reference member. Like that rejection -- and unlike the
// profile-gated pointer/union rules -- this fires regardless of -fprofiles.
template <typename T>
struct DependentRefField {
  T m [[uninit]]; // #dependent-ref-field
};
template struct DependentRefField<long>; // OK
template struct DependentRefField<int &>; // expected-note {{in instantiation of template class 'DependentRefField<int &>' requested here}} \
                                          // no-profiles-note {{in instantiation of template class 'DependentRefField<int &>' requested here}}
// expected-error@#dependent-ref-field {{'uninit' attribute cannot be applied to a reference}}
// no-profiles-error@#dependent-ref-field {{'uninit' attribute cannot be applied to a reference}}

int g_ref_target = 0;
template <typename T>
void dependent_ref_local() {
  T v [[uninit]] = g_ref_target; // #dependent-ref-local
  (void)v;
}
template void dependent_ref_local<int &>(); // expected-note {{in instantiation of function template specialization 'dependent_ref_local<int &>' requested here}} \
                                            // no-profiles-note {{in instantiation of function template specialization 'dependent_ref_local<int &>' requested here}}
// expected-error@#dependent-ref-local {{'uninit' attribute cannot be applied to a reference}}
// no-profiles-error@#dependent-ref-local {{'uninit' attribute cannot be applied to a reference}}

// Suppression on the dependent member carries through instantiation.
template <typename T>
struct DependentFieldSuppressed {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] T m [[uninit]];
};
template struct DependentFieldSuppressed<int *>;

// A member of a union template fires union_marker at instantiation regardless
// of the substituted type (paper section 5.6).
template <typename T>
union DependentUnion {
  T m [[uninit]]; // #dependent-union-member
  int tag;
};
int dependent_union_size = sizeof(DependentUnion<int>); // expected-note {{in instantiation of template class 'DependentUnion<int>' requested here}}
// expected-error@#dependent-union-member {{'[[uninit]]' cannot be applied to a union member under profile 'std::init'}}

// expected-error@+2 {{'uninit' attribute only applies to variables and non-static data members}}
// no-profiles-error@+1 {{'uninit' attribute only applies to variables and non-static data members}}
[[uninit]] void f();

// Subjects on which "leave uninitialized" is meaningless are rejected
// regardless of -fprofiles.
struct ReferenceField {
  int &r [[uninit]]; // expected-error {{'uninit' attribute cannot be applied to a reference}} \
                            // no-profiles-error {{'uninit' attribute cannot be applied to a reference}}
};

void test_invalid_subjects(int p [[uninit]]) { // expected-error {{'uninit' attribute cannot be applied to a function parameter}} \
                                                      // no-profiles-error {{'uninit' attribute cannot be applied to a function parameter}}
  int n = 0;
  int &lr [[uninit]] = n; // expected-error {{'uninit' attribute cannot be applied to a reference}} \
                                 // no-profiles-error {{'uninit' attribute cannot be applied to a reference}}
  int arr[2] = {1, 2};
  [[uninit]] auto [a, b] = arr; // expected-error {{'uninit' attribute cannot be applied to a structured binding}} \
                                       // no-profiles-error {{'uninit' attribute cannot be applied to a structured binding}}
  (void)p; (void)lr; (void)a; (void)b;
}

// The marker re-check on the instantiated field is not suppressed by a
// [[profiles::suppress(std::init)]] live at the point of instantiation --
// the pattern's tokens are outside that dominion (P3589R2 s2.4p3).
template <typename T>
struct LeakMarker {
  T p [[uninit]]; // #leak-marker-field
};
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::init)]] LeakMarker<int *> leak_marker_use{nullptr}; // expected-note {{in instantiation of template class 'LeakMarker<int *>' requested here}}
// expected-error@#leak-marker-field {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}
