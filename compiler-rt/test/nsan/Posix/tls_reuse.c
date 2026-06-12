/// The static TLS block is reused among by threads. The shadow is cleared.
// RUN: %clang_nsan %s -o %t
// RUN: env NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=19 %run %t

#include <pthread.h>
#include <stdio.h>

__thread float x;

static void *ThreadFn(void *a) {
  long i = (long)a;
  for (long j = i * 1000; j < (i + 1) * 1000; j++)
    x += j;
  printf("%f\n", x);
  return 0;
}

int main() {
  pthread_t t;
  for (long i = 0; i < 5; ++i) {
    pthread_create(&t, 0, ThreadFn, (void *)i);
    pthread_join(t, 0);
  }
}
