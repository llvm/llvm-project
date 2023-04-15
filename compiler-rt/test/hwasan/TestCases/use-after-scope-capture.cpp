// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -mllvm -hwasan-use-after-scope --std=c++11 -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch || riscv64-target-arch
// REQUIRES: stable-runtime

#include <functional>

int main() {
  std::function<int()> f;
  {
    volatile int x = 0;
    f = [&x]() __attribute__((noinline)) {
      return x; // BOOM
      // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
      // We cannot assert the line, after the argument promotion pass this crashes
      // in the BOOM line below instead, when the ref gets turned into a value.
      // CHECK: 0x{{.*}} in {{.*}}use-after-scope-capture.cpp
      // CHECK: Cause: stack tag-mismatch
    };
  }
  return f(); // BOOM
}
