// Check that we detect malloc/delete mismatch only if the appropriate flag
// is set.

// RUN: %clangxx_asan -g %s -o %t 2>&1

// Find error and provide malloc context.
// RUN: %env_asan_opts=alloc_dealloc_mismatch=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=ALLOC-STACK

// No error here.
// RUN: %env_asan_opts=alloc_dealloc_mismatch=0 %run %t

// Also works if no malloc context is available.
// RUN: %env_asan_opts=alloc_dealloc_mismatch=1:malloc_context_size=0:fast_unwind_on_malloc=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=alloc_dealloc_mismatch=1:malloc_context_size=0:fast_unwind_on_malloc=1 not %run %t 2>&1 | FileCheck %s
// REQUIRES: stable-runtime
#include <stdlib.h>

static volatile char *x;

int main() {
  x = (char*)malloc(10);
  x[0] = 0;
  delete x;
}
// CHECK: ERROR: AddressSanitizer: alloc-dealloc-mismatch (malloc vs operator delete) on 0x
// CHECK-NEXT: #0{{.*}}operator delete
// CHECK: #{{.*}}main
// CHECK: is located 0 bytes inside of 10-byte region
// CHECK-NEXT: allocated by thread T0 here:
// ALLOC-STACK-NEXT: #0{{.*}}malloc
// ALLOC-STACK: #{{.*}}main
// CHECK: HINT: {{.*}} you may set ASAN_OPTIONS=alloc_dealloc_mismatch=0
