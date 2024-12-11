

// RUN: %clang_cc1 -fsyntax-only -verify -fbounds-safety %s
#include <ptrcheck.h>

void param_with_count(int *__counted_by(len - 2) buf, int len);

void call_param_with_count(void) {
  // expected-error@+1{{negative count value of -2 for 'buf' of type 'int *__single __counted_by(len - 2)' (aka 'int *__single')}}
  param_with_count(0, 0);
  param_with_count(0, 2);
  // expected-error@+1{{passing null to parameter 'buf' of type 'int *__single __counted_by(len - 2)' (aka 'int *__single') with count value of 1 always fails}}
  param_with_count(0, 3);

  // expected-note@+1{{'arr' declared here}}
  int arr[10] = {0};
  param_with_count(arr, 12);
  // expected-error@+1{{negative count value of -2 for 'buf' of type 'int *__single __counted_by(len - 2)' (aka 'int *__single')}}
  param_with_count(arr, 0);
  // expected-error@+1{{passing array 'arr' (which has 10 elements) to parameter 'buf' of type 'int *__single __counted_by(len - 2)' (aka 'int *__single') with count value of 11 always fails}}
  param_with_count(arr, 13);
  param_with_count(arr, 2);
}

void param_with_count_size(int *__counted_by(size * count) buf, int size, int count);

void call_param_with_count_size(void) {
  // expected-error@+1{{negative count value of -1 for 'buf' of type 'int *__single __counted_by(size * count)' (aka 'int *__single')}}
  param_with_count_size(0, -1, 1);
  param_with_count_size(0, 0, 0);
  // expected-error@+1{{passing null to parameter 'buf' of type 'int *__single __counted_by(size * count)' (aka 'int *__single') with count value of 6 always fails}}
  param_with_count_size(0, 3, 2);

  // expected-note@+1{{'arr' declared here}}
  int arr[10] = {0};
  param_with_count_size(arr, 2, 5);
  // expected-error@+1{{negative count value of -1 for 'buf' of type 'int *__single __counted_by(size * count)' (aka 'int *__single')}}
  param_with_count_size(arr, 1, -1);
  // expected-error@+1{{passing array 'arr' (which has 10 elements) to parameter 'buf' of type 'int *__single __counted_by(size * count)' (aka 'int *__single') with count value of 15 always fails}}
  param_with_count_size(arr, 3, 5);
  param_with_count_size(arr, 0, 0);
}

// params sharing count but different expr and the callers
void param_with_shared_size(void *__sized_by(size - 1) buf1, void *__sized_by(size - 2) buf2, int size);

void call_param_with_shared_size(void) {
  // expected-error@+1{{negative size value of -1 for 'buf1' of type 'void *__single __sized_by(size - 1)' (aka 'void *__single')}}
  param_with_shared_size(0, 0, 0);
  // expected-error@+1{{negative size value of -1 for 'buf2' of type 'void *__single __sized_by(size - 2)' (aka 'void *__single')}}
  param_with_shared_size(0, 0, 1);
  // expected-error@+1{{passing null to parameter 'buf1' of type 'void *__single __sized_by(size - 1)' (aka 'void *__single') with size value of 1 always fails}}
  param_with_shared_size(0, 0, 2);
  // expected-error@+1{{passing null to parameter 'buf1' of type 'void *__single __sized_by(size - 1)' (aka 'void *__single') with size value of 9 always fails}}
  param_with_shared_size(0, 0, 10);

  // expected-note@+1{{'arr1' declared here}}
  char arr1[9] = {0};
  char arr2[8] = {0};
  // expected-error@+1{{negative size value of -1 for 'buf1' of type 'void *__single __sized_by(size - 1)' (aka 'void *__single')}}
  param_with_shared_size(arr1, arr2, 0);
  // expected-error@+1{{negative size value of -1 for 'buf2' of type 'void *__single __sized_by(size - 2)' (aka 'void *__single')}}
  param_with_shared_size(arr1, arr2, 1);
  param_with_shared_size(arr1, arr2, 2);
  param_with_shared_size(arr1, arr2, 10);
  // expected-error@+1{{passing array 'arr1' (which has 9 bytes) to parameter 'buf1' of type 'void *__single __sized_by(size - 1)' (aka 'void *__single') with size value of 11 always fails}}
  param_with_shared_size(arr1, arr2, 12);
}

// returns with count and the callers
void *__sized_by(count * size) return_with_count_size(int count, int size);

void call_return_with_count_size(void) {
  // FIXME: rdar://103368466
  void *buf = return_with_count_size(-1, 1);
}
