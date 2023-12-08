// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

void *Thread1(void *x) {
  barrier_wait(&barrier);
  int *p = (int*)x;
  p[0] = 1;
  return NULL;
}

void *Thread2(void *x) {
  char *p = (char*)x;
  p[2] = 1;
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  int *data = new int(42);
  print_address("ptr1=", 1, data);
  print_address("ptr2=", 1, (char*)data + 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, data);
  pthread_create(&t[1], NULL, Thread2, data);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  delete data;
}

// CHECK: ptr1=[[PTR1:0x[0-9,a-f]+]]
// CHECK: ptr2=[[PTR2:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 4 at [[PTR1]] by thread T1:
// CHECK:   Previous write of size 1 at [[PTR2]] by thread T2:
