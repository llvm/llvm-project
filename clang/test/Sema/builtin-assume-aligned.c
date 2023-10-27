// RUN: %clang_cc1 -DSIZE_T_64 -fsyntax-only -Wno-strict-prototypes -triple x86_64-linux -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes -triple i386-freebsd -verify %s

// __builtin_assume_aligned's second parameter is size_t, which may be 32 bits,
// so test differently when size_t is 32 bits and when it is 64 bits.

int test1(int *a) {
  a = __builtin_assume_aligned(a, 32, 0ull);
  return a[0];
}

int test2(int *a) {
  a = __builtin_assume_aligned(a, 32, 0);
  return a[0];
}

int test3(int *a) {
  a = __builtin_assume_aligned(a, 32);
  return a[0];
}

int test4(int *a) {
  a = __builtin_assume_aligned(a, -32); // expected-error {{requested alignment is not a power of 2}}
  return a[0];
}

int test5(int *a, unsigned *b) {
  a = __builtin_assume_aligned(a, 32, b); // expected-error {{incompatible pointer to integer conversion passing 'unsigned int *' to parameter of type}}
  return a[0];
}

int test6(int *a) {
  a = __builtin_assume_aligned(a, 32, 0, 0); // expected-error {{too many arguments to function call, expected at most 3, have 4}}
  return a[0];
}

int test7(int *a) {
  a = __builtin_assume_aligned(a, 31); // expected-error {{requested alignment is not a power of 2}}
  return a[0];
}

int test8(int *a, int j) {
  a = __builtin_assume_aligned(a, j); // expected-error {{must be a constant integer}}
  return a[0];
}

int test9(int *a) {
  a = __builtin_assume_aligned(a, 1ULL << 31); // no-warning
  return a[0];
}

#ifdef SIZE_T_64
int test10(int *a) {
  a = __builtin_assume_aligned(a, 1ULL << 63); // expected-warning {{requested alignment must be 4294967296 bytes or smaller; maximum alignment assumed}}
  return a[0];
}

int test11(int *a) {
  a = __builtin_assume_aligned(a, 1ULL << 32); // no-warning
  return a[0];
}

int test12(int *a) {
  a = __builtin_assume_aligned(a, 1ULL << 33); // expected-warning {{requested alignment must be 4294967296 bytes or smaller; maximum alignment assumed}}
  return a[0];
}
#endif

int test13(int *a) {
  a = (int *)__builtin_assume_aligned(a, 2 * 2.0); // expected-error {{argument to '__builtin_assume_aligned' must be a constant integer}}
  return a[0];
}

int test14(int *a, int b) {
  a = (int *)__builtin_assume_aligned(b, 32); // expected-error {{incompatible integer to pointer conversion passing 'int' to parameter of type 'const void *}}
}

int test15(int *b) {
  int arr[3] = {1, 2, 3};
  b = (int *)__builtin_assume_aligned(arr, 32);
  return b[0];
}

int val(int x) {
  return x;
}

int test16(int *b) {
  b = (int *)__builtin_assume_aligned(val, 32);
  return b[0];
}

void test_void_assume_aligned(void) __attribute__((assume_aligned(32))); // expected-warning {{'assume_aligned' attribute only applies to return values that are pointers}}
int test_int_assume_aligned(void) __attribute__((assume_aligned(16))); // expected-warning {{'assume_aligned' attribute only applies to return values that are pointers}}
void *test_ptr_assume_aligned(void) __attribute__((assume_aligned(64))); // no-warning
void *test_ptr_assume_aligned(void) __attribute__((assume_aligned(8589934592))); // expected-warning {{requested alignment must be 4294967296 bytes or smaller; maximum alignment assumed}}

int j __attribute__((assume_aligned(8))); // expected-warning {{'assume_aligned' attribute only applies to Objective-C methods and functions}}
void *test_no_fn_proto() __attribute__((assume_aligned(32))); // no-warning
void *test_with_fn_proto(void) __attribute__((assume_aligned(128))); // no-warning

void *test_no_fn_proto() __attribute__((assume_aligned(31))); // expected-error {{requested alignment is not a power of 2}}
void *test_no_fn_proto() __attribute__((assume_aligned(32, 73))); // no-warning

void *test_no_fn_proto() __attribute__((assume_aligned)); // expected-error {{'assume_aligned' attribute takes at least 1 argument}}
void *test_no_fn_proto() __attribute__((assume_aligned())); // expected-error {{'assume_aligned' attribute takes at least 1 argument}}
void *test_no_fn_proto() __attribute__((assume_aligned(32, 45, 37))); // expected-error {{'assume_aligned' attribute takes no more than 2 arguments}}
