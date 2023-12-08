// RUN: %clang_hwasan %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

extern void *__hwasan_shadow_memory_dynamic_address;

int main() {
  char *p = (char *)malloc(1);
  free(__hwasan_shadow_memory_dynamic_address);
  // CHECK: ERROR: HWAddressSanitizer: invalid-free on address {{[0x]+}}[[PTR:.*]] at pc {{[0x]+}}[[PC:.*]] on thread T{{[0-9]+}}
  // CHECK: #0 {{[0x]+}}{{.*}}[[PC]] in {{.*}}free
  // CHECK: #1 {{.*}} in main {{.*}}wild-free-shadow.c:[[@LINE-3]]
  // CHECK: {{[0x]+}}{{.*}}[[PTR]] is HWAsan shadow memory.
  // CHECK-NOT: Segmentation fault
  // CHECK-NOT: SIGSEGV
  return 0;
}
