// RUN: %clang_cc1 -triple amdgcn-- -verify -fsyntax-only %s

void func() {
  __private char private_arr[4294967295]; // expected-error {{'__private char[4294967295]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
}

void kernel kernel_func() {
  __private int private_arr[1073741823]; // expected-error {{'__private int[1073741823]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
  __local long local_arr[536870911]; // expected-error {{'__local long[536870911]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
}
