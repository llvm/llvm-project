// RUN: %clangxx_asan -O2 %s --std=c++14 -o %t && %run %t

#include <atomic>
#include <memory>
#include <sanitizer/lsan_interface.h>
#include <thread>
#include <vector>

std::atomic<bool> done;

void foo() {
  std::unique_ptr<char[]> mem;

  while (!done)
    mem.reset(new char[1000000]);
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; ++i)
    threads.emplace_back(foo);

  for (int i = 0; i < 100; ++i)
    __lsan_do_recoverable_leak_check();

  done = true;
  for (auto &t : threads)
    t.join();

  return 0;
}
