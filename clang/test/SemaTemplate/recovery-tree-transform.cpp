// RUN: %clang_cc1 -verify -frecovery-ast %s

template<typename T> int *p = &void(T::error); // expected-error{{cannot take the address of an rvalue}} expected-error{{type 'int' cannot be used prior to '::'}}
int *q = p<int>; // expected-note{{in instantiation of variable template specialization 'p<int>' requested here}}
