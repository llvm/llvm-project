// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"

long long Data;
long long Sync;

void *Thread1(void *x) {
  Data++;
  __atomic_store_n(&Sync, 1, __ATOMIC_RELEASE);
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  return NULL;
}

void *Thread2(void *x) {
  barrier_wait(&barrier);
  if (__atomic_load_n(&Sync, __ATOMIC_RELAXED) != 1)
    exit(0);
  // This store must terminate release sequence of the store in Thread1,
  // thus tsan must detect race between Thread1 and main on Data.
  __atomic_store_n(&Sync, 2, __ATOMIC_RELEASE);
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 3);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  if (__atomic_load_n(&Sync, __ATOMIC_ACQUIRE) != 2)
    exit(0);
  if (Data != 1)
    exit(0);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Read
// CHECK:     #0 main
// CHECK:   Previous write
// CHECK:     #0 Thread1
// CHECK:   Location is global 'Data'
// CHECK: DONE
