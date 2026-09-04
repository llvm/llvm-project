// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

// GH28328: a dependent array bound is not diagnosed in the template pattern;
// the check runs against the instantiated type.
template <int N>
int not_instantiated() {
  int array[N];
  return &array - &array;
}

template <int N>
int instantiated() {
  int array[N];
  return &array - &array; // expected-warning {{subtraction of pointers to type 'int[0]' of zero size has undefined behavior}}
}

int x = instantiated<0>(); // expected-note {{in instantiation of function template specialization 'instantiated<0>' requested here}}
int y = instantiated<1>();
