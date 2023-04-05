// Tests UAF detection where Allocate/Deallocate/Use
// happen in separate threads.
// RUN: %clang_hwasan %s -o %t && not %run %t 2>&1 | FileCheck %s
// REQUIRES: stable-runtime

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#include <sanitizer/hwasan_interface.h>

char *volatile x;
int state;

void *Allocate(void *arg) {
  x = (char*)malloc(10);
  __sync_fetch_and_add(&state, 1);
  while (__sync_fetch_and_add(&state, 0) != 3) {}
  return NULL;
}
void *Deallocate(void *arg) {
  free(x);
  __sync_fetch_and_add(&state, 1);
  while (__sync_fetch_and_add(&state, 0) != 3) {}
  return NULL;
}

void *Use(void *arg) {
  x[5] = 42;
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch on address
  // CHECK: WRITE of size 1 {{.*}} in thread T3
  // CHECK: thread-uaf.c:[[@LINE-3]]
  // CHECK: Cause: use-after-free
  // CHECK: freed by thread T2 here
  // CHECK: in Deallocate
  // CHECK: previously allocated by thread T1 here:
  // CHECK: in Allocate
  // CHECK-DAG: Thread: T2 0x
  // CHECK-DAG: Thread: T3 0x
  // CHECK-DAG: Thread: T0 0x
  // CHECK-DAG: Thread: T1 0x
  __sync_fetch_and_add(&state, 1);
  return NULL;
}

int main() {
  __hwasan_enable_allocator_tagging();
  pthread_t t1, t2, t3;

  pthread_create(&t1, NULL, Allocate, NULL);
  while (__sync_fetch_and_add(&state, 0) != 1) {}
  pthread_create(&t2, NULL, Deallocate, NULL);
  while (__sync_fetch_and_add(&state, 0) != 2) {}
  pthread_create(&t3, NULL, Use, NULL);

  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  pthread_join(t3, NULL);
}
