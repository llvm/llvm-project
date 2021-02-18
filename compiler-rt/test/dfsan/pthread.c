// RUN: %clang_dfsan -mllvm -dfsan-fast-16-labels=true %s -o %t && %run %t

#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <pthread.h>

int volatile x;
int __thread y;

static void *ThreadFn(void *a) {
  y = x;
  assert(dfsan_get_label(y) == 8);
  return 0;
}

int main(void) {
  dfsan_set_label(8, &x, sizeof(x));

  const int kNumThreads = 24;
  pthread_t t[kNumThreads];
  for (size_t i = 0; i < kNumThreads; ++i) {
    pthread_create(&t[i], 0, ThreadFn, (void *)i);
  }
  for (size_t i = 0; i < kNumThreads; ++i) {
    pthread_join(t[i], 0);
  }
  return 0;
}
