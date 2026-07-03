// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics -std=c++11 %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify -fptrauth-intrinsics -std=c++11 %s

template <typename T> struct G {
  T __ptrauth(0,0,1234) test;
  // expected-error@-1 2 {{type '__ptrauth(0,0,1234) T' is already '__ptrauth'-qualified}}
};

template <typename T> struct Indirect {
  G<T> layers;
  // expected-note@-1{{in instantiation of template class 'G<void *__ptrauth(0,0,1235)>' requested here}}
  // expected-note@-2{{in instantiation of template class 'G<void *__ptrauth(0,0,1234)>' requested here}}
};

template <int K, int A, int D>
struct TemplateParameters {
  void * __ptrauth(K, 0, 100) m1; // expected-error {{expression is not an integer constant expression}}
  void * __ptrauth(0, A, 100) m2; // expected-error {{argument to '__ptrauth' must be an integer constant expression}}
  void * __ptrauth(0, 0, D) m3; // expected-error {{argument to '__ptrauth' must be an integer constant expression}}
};

void f3() {
  // FIXME: consider loosening the restrictions so that the first two cases are accepted.
  Indirect<void* __ptrauth(0,0,1234)> one;
  // expected-note@-1{{in instantiation of template class 'Indirect<void *__ptrauth(0,0,1234)>' requested here}}
  Indirect<void* __ptrauth(0,0,1235)> two;
  // expected-note@-1{{in instantiation of template class 'Indirect<void *__ptrauth(0,0,1235)>' requested here}}
  Indirect<void*> three;
}
