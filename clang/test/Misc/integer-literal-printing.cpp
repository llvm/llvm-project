// RUN: %clang_cc1 %s -fsyntax-only -verify

// PR11179
template <short T> class Type1 {};
template <short T> void Function1(Type1<T>& x) {} // expected-note{{candidate function [with T = -42] not viable: expects an l-value for 1st argument}}

template <unsigned short T> class Type2 {};
template <unsigned short T> void Function2(Type2<T>& x) {} // expected-note{{candidate function [with T = 42] not viable: expects an l-value for 1st argument}}

void Function() {
  Function1(Type1<-42>()); // expected-error{{no matching function for call to 'Function1'}}
  Function2(Type2<42>()); // expected-error{{no matching function for call to 'Function2'}}
}
