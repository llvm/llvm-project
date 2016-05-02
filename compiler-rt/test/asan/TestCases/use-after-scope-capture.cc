// RUN: %clangxx_asan -std=c++11 -O1 -mllvm -asan-use-after-scope=1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

#include <functional>

int main() {
  std::function<int()> f;
  {
    int x = 0;
    f = [&x]() {
      return x;  // BOOM
      // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
      // CHECK: #0 0x{{.*}} in {{.*}}use-after-scope-capture.cc:[[@LINE-2]]
    };
  }
  return f();  // BOOM
}
