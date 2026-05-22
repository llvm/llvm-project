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
