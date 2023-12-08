// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t && %run %t 2>&1 | FileCheck %s
#include <stdio.h>
#include <memory>
#include <thread>

int main() {
  int v1 = 0;
  int v2 = 0;
  std::thread t1;
  std::thread t2;

  {
     auto thingy = std::make_shared<int>(42);
     t1 = std::thread([thingy, &v1] { v1 = *thingy; });
     t2 = std::thread([thingy, &v2] { v2 = *thingy; });
  }

  t1.join();
  t2.join();
  printf("%d %d\n", v1, v2);
  // CHECK-NOT: ThreadSanitizer: data race
  // CHECK: 42 42
  return 0;
}
