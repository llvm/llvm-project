// RUN: %clang -pthread %s -o %t %if freebsd %{ -lstdthreads %}
// RUN: %run %t 2>&1 | FileCheck %s

// The threads come from pthread_create, not thrd_create, so that a failure
// here is due to call_once alone; thrd_* has its own test.

// <threads.h> is missing on Darwin and before glibc 2.28.
// UNSUPPORTED: darwin, glibc && !glibc-2.28
// https://github.com/llvm/llvm-project/issues/199585
// UNSUPPORTED: glibc && tsan

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

enum { kNumThreads = 4 };

static once_flag flag = ONCE_FLAG_INIT;
static int init_calls;
static int shared_data;

static void init(void) {
  ++init_calls;
  shared_data = 42;
}

static void *thread_func(void *arg) {
  (void)arg;
  call_once(&flag, init);
  if (shared_data != 42) {
    fprintf(stderr, "shared_data is %d after call_once\n", shared_data);
    abort();
  }
  return NULL;
}

int main(void) {
  pthread_t threads[kNumThreads];
  for (int i = 0; i < kNumThreads; ++i) {
    if (pthread_create(&threads[i], NULL, thread_func, NULL) != 0) {
      fprintf(stderr, "pthread_create failed\n");
      abort();
    }
  }
  for (int i = 0; i < kNumThreads; ++i) {
    if (pthread_join(threads[i], NULL) != 0) {
      fprintf(stderr, "pthread_join failed\n");
      abort();
    }
  }
  if (init_calls != 1) {
    fprintf(stderr, "init ran %d times\n", init_calls);
    abort();
  }
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
