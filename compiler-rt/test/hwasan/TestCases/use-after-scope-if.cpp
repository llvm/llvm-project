// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -mllvm -hwasan-use-after-scope -O1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch

int *p;
bool b = true;

int main() {
  if (b) {
    int x[5];
    p = x + 1;
  }
  return *p; // BOOM
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
  // CHECK:  #0 0x{{.*}} in main {{.*}}.cpp:[[@LINE-2]]
  // CHECK: Cause: stack tag-mismatch
}
