// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
// Race between an aligned access and an unaligned access, which
// touches the same memory region.
#include "test.h"
#include <stdint.h>

uint64_t Global[2];

void *Thread1(void *x) {
  Global[1]++;
  barrier_wait(&barrier);
  return NULL;
}

void *Thread2(void *x) {
  barrier_wait(&barrier);
  char *p1 = reinterpret_cast<char *>(&Global[0]);
  struct __attribute__((packed, aligned(1))) u_uint64_t { uint64_t val; };
  u_uint64_t *p4 = reinterpret_cast<u_uint64_t *>(p1 + 1);
  (*p4).val++;
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "Pass\n");
  // CHECK: ThreadSanitizer: data race
  // CHECK: Pass
  return 0;
}
