// RUN: %clangxx_asan -O2 %s -o %t
// RUN: %env_asan_opts=fast_unwind_on_malloc=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-FAST
// RUN: %env_asan_opts=fast_unwind_on_malloc=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SLOW

// Test how well we unwind in presence of qsort in the stack
// (i.e. if we can unwind through a function compiled w/o frame pointers).
// https://code.google.com/p/address-sanitizer/issues/detail?id=137

// Fast unwinder is only available on x86_64 and i386.
// REQUIRES: x86-target-arch

// REQUIRES: compiler-rt-optimized

#include <stdlib.h>
#include <stdio.h>

int *GlobalPtr;

extern "C" {
int QsortCallback(const void *a, const void *b) {
  char *x = (char*)a;
  char *y = (char*)b;
  printf("Calling QsortCallback\n");
  GlobalPtr = new int[10];
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
  return GlobalPtr[10];
}

// Fast unwind may not unwind through qsort.
// CHECK-FAST: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK-FAST: is located 0 bytes after
// CHECK-FAST: #0{{.*}}operator new
// CHECK-FAST-NEXT: #1{{.*}}QsortCallback

// CHECK-SLOW: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK-SLOW: is located 0 bytes after
// CHECK-SLOW: #0{{.*}}operator new
// CHECK-SLOW-NEXT: #1{{.*}}QsortCallback
// CHECK-SLOW: #{{.*}}MyQsort
// CHECK-SLOW-NEXT: #{{.*}}main
