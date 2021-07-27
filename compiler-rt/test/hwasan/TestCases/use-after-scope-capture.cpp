// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -mllvm -hwasan-use-after-scope --std=c++11 -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch

#include <functional>

int main() {
  std::function<int()> f;
  {
    int x = 0;
    f = [&x]() __attribute__((noinline)) {
      return x; // BOOM
      // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
      // CHECK: #0 0x{{.*}} in {{.*}}use-after-scope-capture.cpp:[[@LINE-2]]
      // CHECK: Cause: stack tag-mismatch
    };
  }
  return f(); // BOOM
}
