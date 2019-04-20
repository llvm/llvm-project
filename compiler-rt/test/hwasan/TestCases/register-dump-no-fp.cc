// RUN: %clangxx_hwasan -fomit-frame-pointer -momit-leaf-frame-pointer \
// RUN:   -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx_hwasan -fomit-frame-pointer -momit-leaf-frame-pointer \
// RUN:   -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx_hwasan -fomit-frame-pointer -momit-leaf-frame-pointer \
// RUN:   -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx_hwasan -fomit-frame-pointer -momit-leaf-frame-pointer \
// RUN:   -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

// This test ensures that the CFA is implemented properly for slow
// (non-frame-pointer) unwinding.
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((noinline)) void f(int *p) { *p = 3; }

// CHECK: ERROR: HWAddressSanitizer:
// CHECK: #0 {{.*}} in f(int*) {{.*}}register-dump-no-fp.cc:[[@LINE-3]]

int main() {
  __hwasan_enable_allocator_tagging();

  int *volatile a = new int;
  a = (int *)__hwasan_tag_pointer(a, 0);
  f(a);
  // CHECK: #1 {{.*}} in main {{.*}}register-dump-no-fp.cc:[[@LINE-1]]
}
