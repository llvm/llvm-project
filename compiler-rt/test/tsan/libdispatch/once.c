// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <dispatch/dispatch.h>

#include "../test.h"

static const long kNumThreads = 4;

long global;
long global2;

static dispatch_once_t once_token;
static dispatch_once_t once_token2;

void f(void *a) {
  global2 = 42;
  usleep(100000);
}

void *Thread(void *a) {
  barrier_wait(&barrier);

  dispatch_once(&once_token, ^{
    global = 42;
    usleep(100000);
  });
  long x = global;

  dispatch_once_f(&once_token2, NULL, f);
  long x2 = global2;

  fprintf(stderr, "global = %ld\n", x);
  fprintf(stderr, "global2 = %ld\n", x2);
  return 0;
}

int main() {
  fprintf(stderr, "Hello world.\n");
  barrier_init(&barrier, kNumThreads);

  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    pthread_create(&t[i], 0, Thread, 0);
  }
  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(t[i], 0);
  }

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
