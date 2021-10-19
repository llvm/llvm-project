// RUN: %clangxx_dfsan %s -fno-exceptions -o %t && %run %t
// RUN: %clangxx_dfsan -DORIGIN_TRACKING -mllvm -dfsan-track-origins=1 %s -fno-exceptions -o %t && %run %t
//
// REQUIRES: x86_64-target-arch
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

struct arg_struct {
  size_t index;
  dfsan_origin origin;
};

static void *ThreadFn(void *arg) {
  if (((arg_struct *)arg)->index % 2) {
    int i = 10;
    dfsan_set_label(8, (void *)&i, sizeof(i));
    atomic_i.store(i, std::memory_order_relaxed);
    return 0;
  }
  int j = atomic_i.load();
  assert(dfsan_get_label(j) == 0 || dfsan_get_label(j) == 2);
#ifdef ORIGIN_TRACKING
  if (dfsan_get_label(j) == 2)
    assert(dfsan_get_init_origin(&j) == ((arg_struct *)arg)->origin);
#endif
  return 0;
}

int main(void) {
  int i = 10;
  dfsan_set_label(2, (void *)&i, sizeof(i));
#ifdef ORIGIN_TRACKING
  dfsan_origin origin = dfsan_get_origin(i);
#endif
  atomic_i.store(i, std::memory_order_relaxed);
  const int kNumThreads = 24;
  pthread_t t[kNumThreads];
  arg_struct args[kNumThreads];
  for (int i = 0; i < kNumThreads; ++i) {
    args[i].index = i;
#ifdef ORIGIN_TRACKING
    args[i].origin = origin;
#endif
    pthread_create(&t[i], 0, ThreadFn, (void *)(args + i));
  }
  for (int i = 0; i < kNumThreads; ++i)
    pthread_join(t[i], 0);
  return 0;
}
