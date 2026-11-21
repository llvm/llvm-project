// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify -x c++ %s

// The mode attribute changes the width of a type, but it must not discard the
// qualifiers written on the original type: 'const int x
// __attribute__((mode(QI)))' still declares a const object.

void const_param(const int x __attribute__((mode(QI)))) { // expected-note {{variable 'x' declared const here}}
  x = 1; // expected-error {{cannot assign to variable 'x' with const-qualified type 'const signed char'}}
}

typedef const int const_qi __attribute__((mode(QI)));

void const_typedef(void) {
  extern const_qi v; // expected-note {{variable 'v' declared const here}}
  v = 1; // expected-error {{cannot assign to variable 'v' with const-qualified type 'const_qi' (aka 'const signed char')}}
}

#ifndef __cplusplus
void volatile_param(volatile int x __attribute__((mode(QI)))) {
  _Static_assert(_Generic(&x, volatile signed char *: 1, default: 0),
                 "volatile qualifier was dropped");
}

void takes_plain_ptr(signed char *); // expected-note {{passing argument to parameter here}}
__attribute__((address_space(1))) int global_as __attribute__((mode(QI)));

void address_space_global(void) {
  takes_plain_ptr(&global_as); // expected-error {{passing '__attribute__((address_space(1))) signed char *' to parameter of type 'signed char *' changes address space of pointer}}
}
#endif
