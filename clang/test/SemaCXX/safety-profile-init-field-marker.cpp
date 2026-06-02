// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

struct PlainField {
  int m [[uninitialized]];
};

struct PlainFieldPrefix {
  [[uninitialized]] int m;
};

struct FieldWithNSDMI {
  int m [[uninitialized]] = 0;
};

struct FieldWithNSDMIPrefix {
  [[uninitialized]] int m = 0;
};

struct WithStaticDataMember {
  static int s [[uninitialized]];
  [[uninitialized]] static int t;
};
int WithStaticDataMember::s;
int WithStaticDataMember::t;

struct MultipleFields {
  int a [[uninitialized]];
  int b = 0;
  int c [[uninitialized]] = 0;
};

template <typename T>
struct DependentField {
  T m [[uninitialized]];
};
template struct DependentField<int>;

// expected-error@+2 {{'uninitialized' attribute only applies to variables and non-static data members}}
// no-profiles-error@+1 {{'uninitialized' attribute only applies to variables and non-static data members}}
[[uninitialized]] void f();

// Subjects on which "leave uninitialized" is meaningless are rejected
// regardless of -fprofiles.
struct ReferenceField {
  int &r [[uninitialized]]; // expected-error {{'uninitialized' attribute cannot be applied to a reference}} \
                            // no-profiles-error {{'uninitialized' attribute cannot be applied to a reference}}
};

void test_invalid_subjects(int p [[uninitialized]]) { // expected-error {{'uninitialized' attribute cannot be applied to a function parameter}} \
                                                      // no-profiles-error {{'uninitialized' attribute cannot be applied to a function parameter}}
  int n = 0;
  int &lr [[uninitialized]] = n; // expected-error {{'uninitialized' attribute cannot be applied to a reference}} \
                                 // no-profiles-error {{'uninitialized' attribute cannot be applied to a reference}}
  int arr[2] = {1, 2};
  [[uninitialized]] auto [a, b] = arr; // expected-error {{'uninitialized' attribute cannot be applied to a structured binding}} \
                                       // no-profiles-error {{'uninitialized' attribute cannot be applied to a structured binding}}
  (void)p; (void)lr; (void)a; (void)b;
}
