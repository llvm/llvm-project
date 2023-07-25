// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -O1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch || riscv64-target-arch

volatile int *p;

int main() {
  // Variable goes in and out of scope.
  for (int i = 0; i < 3; ++i) {
    int x[3] = {i, i, i};
    p = x + i;
  }
  return *p; // BOOM
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope-loop-bug.cpp:[[@LINE-2]]
  // CHECK: Cause: stack tag-mismatch
}
