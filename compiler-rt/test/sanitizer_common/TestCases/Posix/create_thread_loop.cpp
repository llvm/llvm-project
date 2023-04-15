// Simple stress test for of pthread_create. Increase arg to use as benchmark.

// RUN: %clangxx -O3 -pthread %s -o %t && %run %t 1000

#include <pthread.h>
#include <stdlib.h>

extern "C" const char *__asan_default_options() {
  // 32bit asan can allocate just a few FakeStacks.
  return sizeof(void *) < 8 ? "detect_stack_use_after_return=0" : "";
}

static void *null_func(void *args) { return nullptr; }

int main(int argc, char **argv) {
  int n = atoi(argv[1]);
  for (int i = 0; i < n; ++i) {
    pthread_t thread;
    pthread_create(&thread, 0, null_func, NULL);
    pthread_detach(thread);
  }
  return 0;
}
