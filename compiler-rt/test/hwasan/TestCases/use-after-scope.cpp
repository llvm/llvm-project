// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -O1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch || riscv64-target-arch

volatile int *p = 0;

int main() {
  {
    int x = 0;
    p = &x;
  }
  *p = 5; // BOOM
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope.cpp:[[@LINE-2]]
  // CHECK: Cause: stack tag-mismatch
  return 0;
}
