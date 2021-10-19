// RUN: %clang_cc1 -fsyntax-only -verify %s

// return values
void test_void_alloc_align(void) __attribute__((alloc_align(1))); // expected-warning {{'alloc_align' attribute only applies to return values that are pointers}}
void *test_ptr_alloc_align(unsigned long long a) __attribute__((alloc_align(1))); // no-warning

int j __attribute__((alloc_align(1))); // expected-warning {{'alloc_align' attribute only applies to non-K&R-style functions}}
void *test_no_params_zero(void) __attribute__((alloc_align(0))); // expected-error {{'alloc_align' attribute parameter 1 is out of bounds}}
void *test_no_params(void) __attribute__((alloc_align(1))); // expected-error {{'alloc_align' attribute parameter 1 is out of bounds}}
void *test_incorrect_param_type(float a) __attribute__((alloc_align(1))); // expected-error {{'alloc_align' attribute argument may only refer to a function parameter of integer type}}

// argument type
void *test_bad_param_type(void) __attribute((alloc_align(1.1))); // expected-error {{'alloc_align' attribute requires parameter 1 to be an integer constant}}

// argument count
void *test_no_fn_proto(int x, int y) __attribute__((alloc_align)); // expected-error {{'alloc_align' attribute takes one argument}}
void *test_no_fn_proto(int x, int y) __attribute__((alloc_align())); // expected-error {{'alloc_align' attribute takes one argument}}
void *test_no_fn_proto(int x, int y) __attribute__((alloc_align(32, 45, 37))); // expected-error {{'alloc_align' attribute takes one argument}}

void *passthrought(int a) {
  return test_ptr_alloc_align(a);
}
void *align16() {
  return test_ptr_alloc_align(16);
}
void *align15() {
  return test_ptr_alloc_align(15); // expected-warning {{requested alignment is not a power of 2}}
}
void *align1073741824() {
  return test_ptr_alloc_align(8589934592); // expected-warning {{requested alignment must be 4294967296 bytes or smaller; maximum alignment assumed}}
}
