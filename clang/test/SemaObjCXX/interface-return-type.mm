// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

@class NSObject;
template<typename T> struct C {
      static T f(); // expected-error {{interface type 'NSObject' cannot be returned by value; did you forget * in 'NSObject'?}}
};
int g() { NSObject *x = C<NSObject>::f(); }//expected-error {{no member named 'f' in 'C<NSObject>'}} expected-note {{in instantiation of template class 'C<NSObject>' requested here}}
