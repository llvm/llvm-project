// RUN: %clangxx_tsan -std=c++11 %s -o %t
// RUN: %env_tsan_opts=detect_deadlocks=1 %run %t 2>&1 | FileCheck %s

#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

std::mutex m1;
std::mutex m2;
std::mutex m3;
std::atomic<int> counter{0};

void foo() {
  std::unique_lock<std::mutex> lock1(m1);
  std::unique_lock<std::mutex> lock2(m2);
}

void xoo() {
  foo();
  std::unique_lock<std::mutex> lock2(m2);
  std::unique_lock<std::mutex> lock3(m3);
}

void yoo() {
  std::unique_lock<std::mutex> lock3(m3);
  std::unique_lock<std::mutex> lock1(m1);
}

int main() {
  std::thread first([] {
    while (true) {
      xoo();
      if (counter > 10)
        break;
    }
  });
  std::thread second([] {
    while (true) {
      yoo();
      counter++;
      if (counter > 10)
        break;
    }
  });
  first.join();
  second.join();
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer