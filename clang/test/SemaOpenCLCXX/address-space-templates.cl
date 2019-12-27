//RUN: %clang_cc1 %s -cl-std=clc++ -pedantic -verify -fsyntax-only

template <typename T>
struct S {
  T a;        // expected-error{{field may not be qualified with an address space}}
  T f1();     // we ignore address space on a return types.
  void f2(T); // expected-error{{parameter may not be qualified with an address space}}
};

template <typename T>
T foo1(__global T *i) { // expected-note{{candidate template ignored: substitution failure [with T = __local int]: conflicting address space qualifiers are provided between types '__global T' and '__local int'}}
  return *i;
}

template <typename T>
T *foo2(T *i) {
  return i;
}

template <typename T>
void foo3() {
  __private T ii; // expected-error{{conflicting address space qualifiers are provided between types '__private T' and '__global int'}}
}

void bar() {
  S<const __global int> sintgl; // expected-note{{in instantiation of template class 'S<const __global int>' requested here}}

  foo1<__local int>(1); // expected-error{{no matching function for call to 'foo1'}}
  foo2<__global int>(0);
  foo3<__global int>(); // expected-note{{in instantiation of function template specialization 'foo3<__global int>' requested here}}
}
