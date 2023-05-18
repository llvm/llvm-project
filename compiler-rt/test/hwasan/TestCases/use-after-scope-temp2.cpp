// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -std=c++11 -O1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch || riscv64-target-arch

struct IntHolder {
  __attribute__((noinline)) const IntHolder &Self() const {
    return *this;
  }
  int val = 3;
};

const IntHolder *saved;

int main(int argc, char *argv[]) {
  saved = &IntHolder().Self();
  int x = saved->val; // BOOM
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope-temp2.cpp:[[@LINE-2]]
  // CHECK: Cause: stack tag-mismatch
  return x;
}
