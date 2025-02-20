// RUN: %clang_hwasan %s -o %t && not %env_hwasan_opts=verbose_threads=1 %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#include <sanitizer/hwasan_interface.h>

void *BoringThread(void *arg) {
  char * volatile x = (char*)malloc(10);
  x[5] = 0;
  free(x);
  return NULL;
}

// CHECK: Creating  : T0
// CHECK: Creating  : T1
// CHECK: Destroying: T1
// CHECK: Creating  : T1100
// CHECK: Destroying: T1100
// CHECK: Creating  : T1101

void *UAFThread(void *arg) {
  char * volatile x = (char*)malloc(10);
  fprintf(stderr, "ZZZ %p\n", x);
  fflush(stderr);
  free(x);
  x[5] = 42;
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch on address
  // CHECK: WRITE of size 1
  // CHECK: many-threads-uaf.c:[[@LINE-3]]
  // CHECK: Thread: T1101
  return NULL;
}

int main() {
  __hwasan_enable_allocator_tagging();
  pthread_t t;
  for (int i = 0; i < 1100; i++) {
    pthread_create(&t, NULL, BoringThread, NULL);
    pthread_join(t, NULL);
  }
  pthread_create(&t, NULL, UAFThread, NULL);
  pthread_join(t, NULL);
}
