// RUN: %clangxx_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <iostream>
#include <future>
#include <vector>

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  auto my_task = [] { return 42; };

  std::vector<std::thread> threads;

  for (int i = 0; i < 100; i++) {
    std::packaged_task<int(void)> task(my_task);
    std::future<int> future = task.get_future();
    threads.push_back(std::thread(std::move(task)));
  }

  for (auto &t : threads) {
    t.join();
  }

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
