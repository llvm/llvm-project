//RUN: %clang_cc1 %s -cl-std=c++ -pedantic -verify -fsyntax-only

template <typename T>
struct S {
  T a;        // expected-error{{field may not be qualified with an address space}}
  T f1();     // expected-error{{function type may not be qualified with an address space}}
  void f2(T); // expected-error{{parameter may not be qualified with an address space}}
};

void bar() {
  S<const __global int> sintgl; // expected-note{{in instantiation of template class 'S<const __global int>' requested here}}
}
