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

struct WithStaticDataMember {
  static int s [[uninit]];
  [[uninit]] static int t;
};
int WithStaticDataMember::s;
int WithStaticDataMember::t;

struct MultipleFields {
  int a [[uninit]];
  int b = 0;
  int c [[uninit]] = 0; // expected-error {{variable 'c' cannot be both '[[uninit]]' and have an initializer under profile 'std::init'}}
};

template <typename T>
struct DependentField {
  T m [[uninit]];
};
template struct DependentField<int>;

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
