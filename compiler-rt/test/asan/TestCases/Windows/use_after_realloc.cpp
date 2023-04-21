// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <malloc.h>

int main() {
  char *buffer = (char*)realloc(0, 32),
       *stale = buffer;
  buffer = (char*)realloc(buffer, 64);
  // The 'stale' may now point to a free'd memory.
  stale[0] = 42;
// CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK-NEXT: {{#0 .* main .*use_after_realloc.cpp}}:[[@LINE-3]]
// CHECK: [[ADDR]] is located 0 bytes inside of 32-byte region
// CHECK: freed by thread T0 here:
// CHECK-NEXT: {{#0 .* realloc }}
// CHECK-NEXT: {{#1 .* main .*use_after_realloc.cpp}}:[[@LINE-9]]
// CHECK: previously allocated by thread T0 here:
// CHECK-NEXT: {{#0 .* realloc }}
// CHECK-NEXT: {{#1 .* main .*use_after_realloc.cpp}}:[[@LINE-14]]
  free(buffer);
}
