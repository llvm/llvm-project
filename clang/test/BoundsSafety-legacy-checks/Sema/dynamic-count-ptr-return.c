
// TODO: We should get the same diagnostics with/without return_size (rdar://138982703)

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=guarded  %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected -fbounds-safety-bringup-missing-checks=return_size %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=guarded %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected -fbounds-safety-bringup-missing-checks=return_size %s

// guarded-no-diagnostics

#include <ptrcheck.h>
#include <stdint.h>

// expected-note@+1 2{{'g_array' declared here}}
static int32_t g_array[42];

// __counted_by()/__sized_by() with a negative count and any pointer.

int *__counted_by(-1) negative_cb(void) {
  // expected-error@+1{{negative count value of -1 for 'int *__single __counted_by(-1)'}}
  return 0;
}

void *__sized_by(-1) negative_sb(void) {
  // expected-error@+1{{negative size value of -1 for 'void *__single __sized_by(-1)'}}
  return 0;
}

// __counted_by_or_null()/__sized_by_or_null() with a negative count and nonnull ptr.

int32_t *__counted_by_or_null(-1) negative_cbn(void) {
  // expected-error@+1{{negative count value of -1 for 'int32_t *__single __counted_by_or_null(-1)'}}
  return g_array;
}

void *__sized_by_or_null(-1) negative_sbn(void) {
  // expected-error@+1{{negative size value of -1 for 'void *__single __sized_by_or_null(-1)'}}
  return g_array;
}

// __counted_by()/__sized_by() with a positive count and an array.

int32_t *__counted_by(42) array_cb_ok(void) {
  return g_array;
}

void *__sized_by(42*4) array_sb_ok(void) {
  return g_array;
}

int32_t *__counted_by(43) array_cb_bad(void) {
  // expected-error-re@+1{{returning array 'g_array' (which has 42 elements) from a function with result type 'int32_t *__single __counted_by(43)'{{.*}} and count value of 43 always fails}}
  return g_array;
}

void *__sized_by(42*4+1) array_sb_bad(void) {
  // expected-error-re@+1{{returning array 'g_array' (which has 168 bytes) from a function with result type 'void *__single __sized_by(169)'{{.*}} and size value of 169 always fails}}
  return g_array;
}

// __single to __counted_by()/__sized_by() with a positive count/size.

int32_t *__counted_by(1) single_cb_ok(int32_t *__single p) {
  return p;
}

void *__sized_by(4) single_sb_ok(int32_t *__single p) {
  return p;
}

int32_t *__counted_by(2) single_cb_bad(int32_t *__single p) {
  // expected-error-re@+1{{returning 'int32_t *__single'{{.*}} from a function with result type 'int32_t *__single __counted_by(2)'{{.*}} and count value of 2 always fails}}
  return p;
}

void *__sized_by(5) single_sb_bad(int32_t *__single p) {
  // expected-error-re@+1{{returning 'int32_t *__single'{{.*}} with pointee of size 4 from a function with result type 'void *__single __sized_by(5)'{{.*}} and size value of 5 always fails}}
  return p;
}

// NULL to __counted_by()/__sized_by() with a positive count.

int *__counted_by(42) null_cb(int arg) {
  switch (arg) {
    case 0:
      // expected-error@+1{{returning null from a function with result type 'int *__single __counted_by(42)' (aka 'int *__single') and count value of 42 always fails}}
      return (void*) 0;
    case 1:
      // expected-error@+1{{returning null from a function with result type 'int *__single __counted_by(42)' (aka 'int *__single') and count value of 42 always fails}}
      return (int*) 0;
  }

  // expected-error@+1{{returning null from a function with result type 'int *__single __counted_by(42)' (aka 'int *__single') and count value of 42 always fails}}
  return 0;
}

int *__counted_by(size) null_cb_sized(int arg, int size) {
  // No diagnostics
  switch (arg) {
    case 0:
      return 0;
    case 1:
      return (void*) 0;
    case 2:
      return (int*) 0;
  }
  return 0;
}

char *__sized_by(42) null_sb(int arg) {
    switch (arg) {
    case 0:
      // expected-error@+1{{returning null from a function with result type 'char *__single __sized_by(42)' (aka 'char *__single') and size value of 42 always fails}}
      return (void*) 0;
    case 1:
      // expected-error@+1{{returning null from a function with result type 'char *__single __sized_by(42)' (aka 'char *__single') and size value of 42 always fails}}
      return (char*) 0;
  }
  
  // expected-error@+1{{returning null from a function with result type 'char *__single __sized_by(42)' (aka 'char *__single') and size value of 42 always fails}}
  return 0;
}

int *__sized_by(size) null_sb_sized(int arg, int size) {
  // No diagnostics
  switch (arg) {
    case 0:
      return 0;
    case 1:
      return (void*) 0;
    case 2:
      return (int*) 0;
  }
  return 0;
}


// The __counted_by_or_null()/__sized_by_or_null() pointer is set to some unknown value with a negative count/size.

int *__counted_by_or_null(-1) negative_cbn_unknown(int *p) {
  // expected-error@+1{{possibly returning non-null from a function with result type 'int *__single __counted_by_or_null(-1)' (aka 'int *__single') with count value of -1; explicitly return null to remove this warning}}
  return p;
}

void *__sized_by_or_null(-1) negative_sbn_unknown(int *p) {
  // expected-error@+1{{possibly returning non-null from a function with result type 'void *__single __sized_by_or_null(-1)' (aka 'void *__single') with size value of -1; explicitly return null to remove this warning}}
  return p;
}
