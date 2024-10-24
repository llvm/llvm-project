// RUN: %clang_cc1 -verify %s

#define NODEREF __attribute__((noderef))

template <typename T>
int func(T NODEREF *a) { // expected-note 3 {{a declared here}}
  return *a + 1;         // expected-warning 3 {{dereferencing a; was declared with a 'noderef' type}}
}

void func() {
  float NODEREF *f;
  int NODEREF *i;
  func(f); // expected-note{{in instantiation of function template specialization 'func<float>' requested here}}
  func(i); // expected-note{{in instantiation of function template specialization 'func<int>' requested here}}
}
