// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O3 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s < %t.out

#include <sanitizer/msan_interface.h>

int main(int argc, char **argv) {
  int x;
  int *volatile p = &x;
  return __sanitizer_unaligned_load32(p);
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*unaligned_read_origin.cpp:}}[[@LINE-2]]
  // CHECK: Uninitialized value was created by an allocation of 'x' in the stack frame of function 'main'
  // CHECK: {{#0 0x.* in main .*unaligned_read_origin.cpp:}}[[@LINE-7]]
}
