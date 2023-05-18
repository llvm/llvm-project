// Simple stress test for of pthread_create. Increase arg to use as benchmark.

// RUN: %clangxx -O3 -pthread %s -o %t && %run %t 10

// Crashes on Android.
// UNSUPPORTED: android

#include <cstdint>
#include <pthread.h>
#include <stdlib.h>

extern "C" const char *__asan_default_options() {
  // 32bit asan can allocate just a few FakeStacks.
  return sizeof(void *) < 8 ? "detect_stack_use_after_return=0" : "";
}

static void *null_func(void *args) { return nullptr; }

static void *loop(void *args) {
  uintptr_t n = (uintptr_t)args;
  for (int i = 0; i < n; ++i) {
    pthread_t thread;
    if (pthread_create(&thread, 0, null_func, nullptr) == 0)
      pthread_detach(thread);
  }
  return nullptr;
}

int main(int argc, char **argv) {
  uintptr_t n = atoi(argv[1]);
  pthread_t threads[64];
  for (auto &thread : threads)
    while (pthread_create(&thread, 0, loop, (void *)n) != 0) {
    }

  for (auto &thread : threads)
    pthread_join(thread, nullptr);
  return 0;
}
