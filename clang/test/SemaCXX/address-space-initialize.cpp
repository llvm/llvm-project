// RUN: %clang_cc1 -fsyntax-only -verify %s

__attribute__((address_space(42)))
const float withc = 1.0f;

__attribute__((address_space(42)))
volatile float withv = 1.0f;

__attribute__((address_space(42)))
float nocv = 1.0f;

__attribute__((address_space(42)))
float nocv_array[10] = { 1.0f };

__attribute__((address_space(42)))
int nocv_iarray[10] = { 4 };


__attribute__((address_space(9999)))
int* as_ptr = nocv_iarray; // expected-error{{cannot initialize a variable of type '__attribute__((address_space(9999))) int *' with an lvalue of type '__attribute__((address_space(42))) int[10]'}}


__attribute__((address_space(42))) int* __attribute__((address_space(42))) ptr_in_same_addr_space = nocv_iarray;
__attribute__((address_space(42))) int* __attribute__((address_space(999))) ptr_in_different_addr_space = nocv_iarray;

