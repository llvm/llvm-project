// Check that concurent CurrentThreadContext does not crash.
// RUN: %clangxx_lsan -O3 -pthread %s -o %t && %run %t 100

// REQUIRES: lsan-standalone

#include <pthread.h>
#include <stdlib.h>
#include <vector>

#include <sanitizer/common_interface_defs.h>

namespace __lsan {
class ThreadContextLsanBase *GetCurrentThread();
}

void *null_func(void *args) {
  for (int i = 0; i < 100000; ++i)
    __lsan::GetCurrentThread();
  return nullptr;
}

int main(int argc, char **argv) {
  std::vector<pthread_t> threads;
  for (int i = 0; i < atoi(argv[1]); ++i) {
    threads.resize(10);
    for (auto &thread : threads)
      pthread_create(&thread, 0, null_func, NULL);

    for (auto &thread : threads)
      pthread_join(thread, nullptr);
  }
  return 0;
}
