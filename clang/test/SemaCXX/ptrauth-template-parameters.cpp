// RUN: %clang_cc1 -triple arm64e-apple-ios -fsyntax-only -verify -fptrauth-intrinsics -std=c++20 %s

template <typename T> struct G {
  T __ptrauth(0,0,1234) test;
  // expected-error@-1 2 {{type '__ptrauth(0,0,1234) T' is already __ptrauth-qualified}}
};

template <typename T> struct Indirect {
  G<T> layers;
  // expected-note@-1{{in instantiation of template class 'G<void *__ptrauth(0,0,1235)>' requested here}}
  // expected-note@-2{{in instantiation of template class 'G<void *__ptrauth(0,0,1234)>' requested here}}
};

void f3() {
  Indirect<void* __ptrauth(0,0,1234)> one;
  // expected-note@-1{{in instantiation of template class 'Indirect<void *__ptrauth(0,0,1234)>' requested here}}
  Indirect<void* __ptrauth(0,0,1235)> two;
  // expected-note@-1{{in instantiation of template class 'Indirect<void *__ptrauth(0,0,1235)>' requested here}}
  Indirect<void*> three;
}
