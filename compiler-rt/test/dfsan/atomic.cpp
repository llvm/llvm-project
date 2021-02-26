// RUN: %clangxx_dfsan -mllvm -dfsan-fast-16-labels=true %s -fno-exceptions -o %t && %run %t
//
// Use -fno-exceptions to turn off exceptions to avoid instrumenting
// __cxa_begin_catch, std::terminate and __gxx_personality_v0.
//
// TODO: Support builtin atomics. For example, https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
// DFSan instrumentation pass cannot identify builtin callsites yet.

#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <atomic>
#include <pthread.h>

std::atomic<int> atomic_i{0};

static void *ThreadFn(void *arg) {
  if ((size_t)arg % 2) {
    int i = 10;
    dfsan_set_label(8, (void *)&i, sizeof(i));
    atomic_i.store(i, std::memory_order_relaxed);

    return 0;
  }
  int j = atomic_i.load();
  assert(dfsan_get_label(j) == 0 || dfsan_get_label(j) == 2);

  return 0;
}

int main(void) {
  int i = 10;
  dfsan_set_label(2, (void *)&i, sizeof(i));
  atomic_i.store(i, std::memory_order_relaxed);
  const int kNumThreads = 24;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; ++i) {
    pthread_create(&t[i], 0, ThreadFn, (void *)i);
  }
  for (int i = 0; i < kNumThreads; ++i) {
    pthread_join(t[i], 0);
  }
  return 0;
}
