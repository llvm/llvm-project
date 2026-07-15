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
  int m [[uninit]] = 0; // expected-error {{member 'm' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
};

struct FieldWithNSDMIPrefix {
  [[uninit]] int m = 0; // expected-error {{member 'm' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
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
  int c [[uninit]] = 0; // expected-error {{member 'c' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
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

// std::init / uninit_with_initializer, field flavor (paper §4.2 rule 2,
// §5.3): [[uninit]] on a member whose type's default-initialization is not a
// genuine no-op is a contradiction -- something is initialized, or nothing
// is left uninitialized. Diagnosed at the marker, with the reason at the
// member type.
namespace std { enum class byte : unsigned char {}; }

struct RunsCtor { RunsCtor() : cap(0) {} int cap; };
struct TrivialAgg { int a; };

struct MemberRunsCtor {
  RunsCtor s [[uninit]]; // expected-error {{member 's' cannot be marked '[[uninit]]' under profile 'std::init'; default-initialization of its type 'RunsCtor' does not leave it uninitialized}} \
                         // expected-note {{default-initialization of 'RunsCtor' runs a constructor}}
};

struct MemberVacuousKinds {
  int x [[uninit]];        // OK: a scalar member really is left uninitialized
  TrivialAgg t [[uninit]]; // OK: trivial aggregate, a genuine no-op
  std::byte b [[uninit]];  // OK: std::byte may stay uninitialized (paper §4)
};

// An NSDMI'd marked member is the NSDMI flavor's to diagnose -- exactly one
// diagnostic, no field-flavor double (hasInClassInitializer is style-based,
// so the skip holds even while the NSDMI is late-parse-pending).
struct MemberNSDMIOnce {
  RunsCtor s [[uninit]] = RunsCtor(); // expected-error {{member 's' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
};

// A union- or pointer-typed marked member draws only union_marker /
// pointer_marker -- no field-flavor pile-on. Load-bearing for the union: V's
// implicitly deleted default constructor is unusable, so without the
// base-element-type skip the member would draw both diagnostics.
union V { RunsCtor s; };
struct MemberUnionOnly {
  V v [[uninit]]; // expected-error {{'[[uninit]]' cannot be applied to a data member of union type under profile 'std::init'}}
};
struct MemberPointerOnly {
  int *p [[uninit]]; // expected-error {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}
};

// A member type with a deleted (or absent) default constructor can never be
// left default-initialized, so the marker is unsatisfiable.
struct NoDefault { NoDefault() = delete; int x; };
struct MemberDeletedCtor {
  NoDefault n [[uninit]]; // expected-error {{member 'n' cannot be marked '[[uninit]]' under profile 'std::init'; default-initialization of its type 'NoDefault' does not leave it uninitialized}} \
                          // expected-note {{'NoDefault' has no usable default constructor}}
};

// An all-determinate type leaves nothing to acknowledge.
struct Empty {};
struct MemberEmpty {
  Empty e [[uninit]]; // expected-error {{member 'e' cannot be marked '[[uninit]]' under profile 'std::init'; default-initialization of its type 'Empty' does not leave it uninitialized}} \
                      // expected-note {{no subobject of 'Empty' is left uninitialized}}
};

// A written member-initializer in some constructor does not rescue the
// marker: both branches contradict it (the member is initialized either
// way).
struct MemberCtorInit {
  RunsCtor s [[uninit]]; // expected-error {{member 's' cannot be marked '[[uninit]]' under profile 'std::init'; default-initialization of its type 'RunsCtor' does not leave it uninitialized}} \
                         // expected-note {{default-initialization of 'RunsCtor' runs a constructor}}
  MemberCtorInit() : s{} {}
};

// Arrays key on the base element type, like the other marker rules.
struct MemberArrayOfCtor {
  RunsCtor arr [[uninit]][2]; // expected-error {{member 'arr' cannot be marked '[[uninit]]' under profile 'std::init'; default-initialization of its type 'RunsCtor[2]' does not leave it uninitialized}} \
                              // expected-note {{default-initialization of 'RunsCtor' runs a constructor}}
};

// A dependent marked member defers on the pattern and fires once at
// instantiation, when the substituted type's vacuity is known.
template <typename T>
struct DependentVacuity {
  T m [[uninit]]; // #dependent-vacuity-member
};
template struct DependentVacuity<long>; // OK: vacuous for a scalar
template struct DependentVacuity<RunsCtor>; // expected-note {{in instantiation of template class 'DependentVacuity<RunsCtor>' requested here}}
// expected-error@#dependent-vacuity-member {{member 'm' cannot be marked '[[uninit]]' under profile 'std::init'; default-initialization of its type 'RunsCtor' does not leave it uninitialized}}
// expected-note@#dependent-vacuity-member {{default-initialization of 'RunsCtor' runs a constructor}}

// Suppression: rule-targeted on the field, and whole-profile on the class.
struct SuppressedFieldMarker {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_with_initializer")]] RunsCtor s [[uninit]]; // OK: suppressed
};
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] SuppressedClassFieldMarker {
  RunsCtor s [[uninit]]; // OK: suppressed by the class-level attribute
};
