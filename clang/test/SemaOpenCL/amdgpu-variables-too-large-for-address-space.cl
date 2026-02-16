// RUN: %clang_cc1 -triple amdgcn-- -verify -fsyntax-only %s

void func() {
  __private char max_size_private_arr[4294967295];
  __private char too_large_private_arr[4294967296]; // expected-error {{'__private char[4294967296]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
}

void kernel kernel_func() {
  __private int max_size_private_arr[1073741823];
  __local long max_size_local_arr[536870911];
  __private int too_large_private_arr[1073741824]; // expected-error {{'__private int[1073741824]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
  __local long too_large_local_arr[536870912]; // expected-error {{'__local long[536870912]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
}
