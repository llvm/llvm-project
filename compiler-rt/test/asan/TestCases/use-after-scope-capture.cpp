// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <functional>

int main() {
  std::function<int()> f;
  {
    int x = 0;
    f = [&x]() __attribute__((noinline)) {
      return x;  // BOOM
      // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
      // CHECK: #0 0x{{.*}} in {{.*}}use-after-scope-capture.cpp:[[@LINE-2]]
    };
  }
  return f();  // BOOM
}
