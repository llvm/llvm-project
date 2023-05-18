// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <functional>

int main() {
  std::function<int()> f;
  {
    int x = 0;
    f = [&x]() __attribute__((noinline)) {
      return x;  // BOOM
      // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
      // We cannot assert the line, after the argument promotion pass this crashes
      // in the BOOM line below instead, when the ref gets turned into a value.
      // CHECK: #0 0x{{.*}} in {{.*}}use-after-scope-capture.cpp
    };
  }
  return f();  // BOOM
}
