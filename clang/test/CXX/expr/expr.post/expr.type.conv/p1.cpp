// RUN: %clang_cc1 -std=c++1z -verify %s

template<typename T> struct A { // expected-note 2{{candidate}} expected-note 2{{implicit deduction guide}}
  T t, u;
};
template<typename T> A(T, T) -> A<T>; // expected-note {{deduced conflicting types for parameter 'T'}}
template<typename T> A(A<T>) -> A<T>; // expected-note {{requires 1 argument, but 2 were provided}}

A a = A{1, 2};
A b = A{3, 4.0}; // expected-error {{no viable constructor or deduction guide}}
