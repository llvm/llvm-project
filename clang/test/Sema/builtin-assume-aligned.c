// RUN: %clang_cc1 -fsyntax-only -verify %s

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
  // FIXME: The line below produces {{requested alignment is not a power of 2}}
  // on i386-freebsd, but not on x86_64-linux (for example).
  // a = __builtin_assume_aligned(a, 1ULL << 63);
  return a[0];
}

int test5(int *a, unsigned *b) {
  a = __builtin_assume_aligned(a, 32, b); // expected-warning {{incompatible pointer to integer conversion passing 'unsigned int *' to parameter of type}}
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

int pr43638(int *a) {
  a = __builtin_assume_aligned(a, 8589934592); // expected-warning {{requested alignment must be 4294967296 bytes or smaller; maximum alignment assumed}}
  return a[0];
}
