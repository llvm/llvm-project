// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// Regression test for https://github.com/golang/go/issues/39186

// pthread barriers are not available on OS X
// UNSUPPORTED: darwin

#include "java.h"
#include <string.h>

struct Heap {
  uint64_t data;
  uint64_t ready;
  uint64_t finalized;
  uint64_t wg;
  pthread_barrier_t barrier_finalizer;
  pthread_barrier_t barrier_ballast;
};

void *Thread1(void *p) {
  Heap* heap = (Heap*)p;
  pthread_barrier_wait(&heap->barrier_finalizer);
  __tsan_java_finalize();
  __atomic_fetch_add(&heap->wg, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&heap->finalized, 1, __ATOMIC_RELAXED);
  return 0;
}

void *Thread2(void *p) {
  Heap* heap = (Heap*)p;
  pthread_barrier_wait(&heap->barrier_finalizer);
  heap->data = 1;
  __atomic_store_n(&heap->ready, 1, __ATOMIC_RELEASE);
  return 0;
}

void *Thread3(void *p) {
  Heap* heap = (Heap*)p;
  pthread_barrier_wait(&heap->barrier_finalizer);
  while (__atomic_load_n(&heap->ready, __ATOMIC_ACQUIRE) != 1)
    pthread_yield();
  while (__atomic_load_n(&heap->finalized, __ATOMIC_RELAXED) != 1)
    pthread_yield();
  __atomic_fetch_add(&heap->wg, 1, __ATOMIC_RELEASE);
  return 0;
}

void *Ballast(void *p) {
  Heap* heap = (Heap*)p;
  pthread_barrier_wait(&heap->barrier_ballast);
  return 0;
}

int main() {
  Heap* heap = (Heap*)calloc(sizeof(Heap), 2) + 1;
  __tsan_java_init((jptr)heap, sizeof(*heap));
  __tsan_java_alloc((jptr)heap, sizeof(*heap));
  // Ballast threads merely make the bug a bit easier to trigger.
  const int kBallastThreads = 100;
  pthread_barrier_init(&heap->barrier_finalizer, 0, 4);
  pthread_barrier_init(&heap->barrier_ballast, 0, kBallastThreads + 1);
  pthread_t th[3];
  pthread_create(&th[0], 0, Thread1, heap);
  pthread_create(&th[1], 0, Thread2, heap);
  pthread_t ballast[kBallastThreads];
  for (int i = 0; i < kBallastThreads; i++)
    pthread_create(&ballast[i], 0, Ballast, heap);
  pthread_create(&th[2], 0, Thread3, heap);
  pthread_barrier_wait(&heap->barrier_ballast);
  for (int i = 0; i < kBallastThreads; i++)
    pthread_join(ballast[i], 0);
  pthread_barrier_wait(&heap->barrier_finalizer);
  while (__atomic_load_n(&heap->wg, __ATOMIC_ACQUIRE) != 2)
    pthread_yield();
  if (heap->data != 1)
    exit(printf("no data\n"));
  for (int i = 0; i < 3; i++)
    pthread_join(th[i], 0);
  pthread_barrier_destroy(&heap->barrier_ballast);
  pthread_barrier_destroy(&heap->barrier_finalizer);
  __tsan_java_free((jptr)heap, sizeof(*heap));
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
