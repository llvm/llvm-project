

// RUN: %clang_cc1 -fsyntax-only -verify -fbounds-safety %s
#include <ptrcheck.h>


void test_counted_by(void) {
  int len;
  int *__counted_by(len - 2) buf;
  // expected-error@-1{{negative count value of -2 for 'buf' of type 'int *__single __counted_by(len - 2)' (aka 'int *__single')}}

  int len2 = 2;
  int *__counted_by(len2 - 2) buf2;

  int len3 = 0;
  int *__counted_by(len3 - 2) buf3;
  // expected-error@-1{{negative count value of -2 for 'buf3' of type 'int *__single __counted_by(len3 - 2)' (aka 'int *__single')}}

  int len4 = 0;
  int *__counted_by(len4 + 2) buf4;
  // expected-error@-1{{implicitly initializing 'buf4' of type 'int *__single __counted_by(len4 + 2)' (aka 'int *__single') and count value of 2 with null always fails}}

  int len5 = 0;
  int *__counted_by(len5 + 2) buf5 = 0;
  // expected-error@-1{{initializing 'buf5' of type 'int *__single __counted_by(len5 + 2)' (aka 'int *__single') and count value of 2 with null always fails}}

  int len5_1;
  int *__counted_by(len5_1 + 2) buf5_1 = 0;
  // expected-error@-1{{initializing 'buf5_1' of type 'int *__single __counted_by(len5_1 + 2)' (aka 'int *__single') and count value of 2 with null always fails}}

  int *len6;
  int *__counted_by(*len6) buf6;
  // expected-error@-1{{dereference operator in '__counted_by' is only allowed for function parameters}}

  int *len6_1;
  int *__counted_by(*len6_1 + 2) buf6_1;
  // expected-error@-1{{invalid argument expression to bounds attribute}}

  int len7_1;
  int len7_2;
  int *__counted_by(len7_1 * 2 + len7_2 * 4) buf7;

  int len8_1;
  int len8_2;
  int *__counted_by(len8_1 * 2 + len8_2 * 4 + 1) buf8;
  // expected-error@-1{{implicitly initializing 'buf8' of type 'int *__single __counted_by(len8_1 * 2 + len8_2 * 4 + 1)' (aka 'int *__single') and count value of 1 with null always fails}}

  int len9_1 = 1;
  int len9_2 = 3;
  int *__counted_by(len9_1 * len9_2) buf9;
  // expected-error@-1{{implicitly initializing 'buf9' of type 'int *__single __counted_by(len9_1 * len9_2)' (aka 'int *__single') and count value of 3 with null always fails}}

  int len10_1;
  int len10_2 = 2;
  int *__counted_by(len10_1 * len10_2) buf10;

  int len11_1;
  int len11_2 = 2;
  int *__counted_by(len11_1 * len11_2 + 1) buf11;
  // expected-error@-1{{implicitly initializing 'buf11' of type 'int *__single __counted_by(len11_1 * len11_2 + 1)' (aka 'int *__single') and count value of 1 with null always fails}}

  // don't complain about extern variables
  extern int ext_len;
  extern int *__counted_by(ext_len - 2) ext_buf;
}

void test_counted_by_or_null(void) {
  int arr[2] = {1,2}; // expected-note{{'arr' declared here}}
  int len;
  int *__counted_by_or_null(len - 2) buf = arr;
  // expected-error@-1{{negative count value of -2 for 'buf' of type 'int *__single __counted_by_or_null(len - 2)' (aka 'int *__single')}}

  int len2 = 2;
  int *__counted_by_or_null(len2 - 2) buf2 = arr;

  int len3 = 0;
  int *__counted_by_or_null(len3 + 2) buf3 = arr;

  int *len4;
  int *__counted_by_or_null(*len4) buf4 = arr;
  // expected-error@-1{{dereference operator in '__counted_by_or_null' is only allowed for function parameters}}

  int *len4_1;
  int *__counted_by_or_null(*len4_1 + 2) buf4_1 = arr;
  // expected-error@-1{{invalid argument expression to bounds attribute}}

  int len5_1;
  int len5_2;
  int *__counted_by_or_null(len5_1 * 2 + len5_2 * 4) buf5 = arr;
  // expected-warning@-1{{possibly initializing 'buf5' of type 'int *__single __counted_by_or_null(len5_1 * 2 + len5_2 * 4)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}

  int len6_1;
  int len6_2;
  int *__counted_by_or_null(len6_1 * 2 + len6_2 * 4 - 1) buf6 = arr;
  // expected-error@-1{{negative count value of -1 for 'buf6' of type 'int *__single __counted_by_or_null(len6_1 * 2 + len6_2 * 4 - 1)' (aka 'int *__single')}}

  int len7_1 = 1;
  int len7_2 = 3;
  int *__counted_by_or_null(len7_1 * len7_2) buf7 = arr;
  // expected-error@-1{{initializing 'buf7' of type 'int *__single __counted_by_or_null(len7_1 * len7_2)' (aka 'int *__single') and count value of 3 with array 'arr' (which has 2 elements) always fails}}

  int len8_1;
  int len8_2 = 2;
  int *__counted_by_or_null(len8_1 * len8_2) buf8 = arr;
  // expected-warning@-1{{possibly initializing 'buf8' of type 'int *__single __counted_by_or_null(len8_1 * len8_2)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}

  int len9_1;
  int len9_2 = 2;
  int *__counted_by_or_null(len9_1 * len9_2 + 1) buf9 = arr;
}

void test_sized_by(void) {
  int len;
  void *__sized_by(len - 2) buf;
  // expected-error@-1{{negative size value of -2 for 'buf' of type 'void *__single __sized_by(len - 2)' (aka 'void *__single')}}

  int len2 = 2;
  void *__sized_by(len2 - 2) buf2;

  int len3 = 0;
  void *__sized_by(len3 - 2) buf3;
  // expected-error@-1{{negative size value of -2 for 'buf3' of type 'void *__single __sized_by(len3 - 2)' (aka 'void *__single')}}

  int len4 = 0;
  void *__sized_by(len4 + 2) buf4;
  // expected-error@-1{{implicitly initializing 'buf4' of type 'void *__single __sized_by(len4 + 2)' (aka 'void *__single') and size value of 2 with null always fails}}

  int len5 = 0;
  void *__sized_by(len5 + 2) buf5 = 0;
  // expected-error@-1{{initializing 'buf5' of type 'void *__single __sized_by(len5 + 2)' (aka 'void *__single') and size value of 2 with null always fails}}

  int len5_1;
  void *__sized_by(len5_1 + 2) buf5_1 = 0;
  // expected-error@-1{{initializing 'buf5_1' of type 'void *__single __sized_by(len5_1 + 2)' (aka 'void *__single') and size value of 2 with null always fails}}

  int *len6;
  void *__sized_by(*len6) buf6;
  // expected-error@-1{{dereference operator in '__sized_by' is only allowed for function parameters}}

  int *len6_1;
  void *__sized_by(*len6_1 + 2) buf6_1;
  // expected-error@-1{{invalid argument expression to bounds attribute}}

  int len7_1;
  int len7_2;
  void *__sized_by(len7_1 * 2 + len7_2 * 4) buf7;

  int len8_1;
  int len8_2;
  void *__sized_by(len8_1 * 2 + len8_2 * 4 + 1) buf8;
  // expected-error@-1{{implicitly initializing 'buf8' of type 'void *__single __sized_by(len8_1 * 2 + len8_2 * 4 + 1)' (aka 'void *__single') and size value of 1 with null always fails}}

  int len9_1 = 1;
  int len9_2 = 3;
  void *__sized_by(len9_1 * len9_2) buf9;
  // expected-error@-1{{implicitly initializing 'buf9' of type 'void *__single __sized_by(len9_1 * len9_2)' (aka 'void *__single') and size value of 3 with null always fails}}

  int len10_1;
  int len10_2 = 2;
  void *__sized_by(len10_1 * len10_2) buf10;

  int len11_1;
  int len11_2 = 2;
  void *__sized_by(len11_1 * len11_2 + 1) buf11;
  // expected-error@-1{{implicitly initializing 'buf11' of type 'void *__single __sized_by(len11_1 * len11_2 + 1)' (aka 'void *__single') and size value of 1 with null always fails}}

  // don't complain about extern variables
  extern int ext_len2;
  extern void *__sized_by(ext_len2 - 2) ext_buf2;
}

void test_sized_by_or_null(void) {
  int arr[2] = {1,2};
  int len;
  int *__sized_by_or_null(len - 2) buf = arr;
  // expected-error@-1{{negative size value of -2 for 'buf' of type 'int *__single __sized_by_or_null(len - 2)' (aka 'int *__single')}}

  int len2 = 2;
  int *__sized_by_or_null(len2 - 2) buf2 = arr;

  int len3 = 0;
  int *__sized_by_or_null(len3 + 2) buf3 = arr;

  int *len4;
  int *__sized_by_or_null(*len4) buf4 = arr;
  // expected-error@-1{{dereference operator in '__sized_by_or_null' is only allowed for function parameters}}

  int *len4_1;
  int *__sized_by_or_null(*len4_1 + 2) buf4_1 = arr;
  // expected-error@-1{{invalid argument expression to bounds attribute}}

  int len5_1;
  int len5_2;
  int *__sized_by_or_null(len5_1 * 2 + len5_2 * 4) buf5 = arr;
  // expected-warning@-1{{possibly initializing 'buf5' of type 'int *__single __sized_by_or_null(len5_1 * 2 + len5_2 * 4)' (aka 'int *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}

  int len6_1;
  int len6_2;
  int *__sized_by_or_null(len6_1 * 2 + len6_2 * 4 - 1) buf6 = arr;
  // expected-error@-1{{negative size value of -1 for 'buf6' of type 'int *__single __sized_by_or_null(len6_1 * 2 + len6_2 * 4 - 1)' (aka 'int *__single')}}

  int len7_1 = 1;
  int len7_2 = 3;
  int *__sized_by_or_null(len7_1 * len7_2) buf7 = arr;

  int len8_1;
  int len8_2 = 2;
  int *__sized_by_or_null(len8_1 * len8_2) buf8 = arr;
  // expected-warning@-1{{possibly initializing 'buf8' of type 'int *__single __sized_by_or_null(len8_1 * len8_2)' (aka 'int *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}

  int len9_1;
  int len9_2 = 2;
  int *__sized_by_or_null(len9_1 * len9_2 + 1) buf9 = arr;
}

