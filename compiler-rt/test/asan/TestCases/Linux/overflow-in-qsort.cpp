// RUN: %clangxx_asan -O2 %s -o %t
// RUN: %env_asan_opts=fast_unwind_on_fatal=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-FAST
// RUN: %env_asan_opts=fast_unwind_on_fatal=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SLOW

// Test how well we unwind in presence of qsort in the stack
// (i.e. if we can unwind through a function compiled w/o frame pointers).
// https://code.google.com/p/address-sanitizer/issues/detail?id=137

// Fast unwinder is only available on x86_64 and i386.
// REQUIRES: x86-target-arch

#include <stdlib.h>
#include <stdio.h>

int global_array[10];
volatile int one = 1;

extern "C" {
int QsortCallback(const void *a, const void *b) {
  char *x = (char*)a;
  char *y = (char*)b;
  printf("Calling QsortCallback\n");
  global_array[one * 10] = 0;  // BOOM
  return (int)*x - (int)*y;
}

__attribute__((noinline))
void MyQsort(char *a, size_t size) {
  printf("Calling qsort\n");
  qsort(a, size, sizeof(char), QsortCallback);
  printf("Done\n");  // Avoid tail call.
}
}  // extern "C"

int main() {
  char a[2] = {1, 2};
  MyQsort(a, 2);
}

// Fast unwind may not unwind through qsort.
// CHECK-FAST: ERROR: AddressSanitizer: global-buffer-overflow
// CHECK-FAST: #0{{.*}} in QsortCallback
// CHECK-FAST: is located 0 bytes after global variable 'global_array

// CHECK-SLOW: ERROR: AddressSanitizer: global-buffer-overflow
// CHECK-SLOW: #0{{.*}} in QsortCallback
// CHECK-SLOW: #{{.*}} in MyQsort
// CHECK-SLOW: #{{.*}} in main
// CHECK-SLOW: is located 0 bytes after global variable 'global_array
