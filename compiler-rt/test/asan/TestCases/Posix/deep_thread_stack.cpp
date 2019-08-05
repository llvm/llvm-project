// RUN: %clangxx_asan -O0 %s -pthread -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -pthread -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -pthread -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -pthread -o %t && not %run %t 2>&1 | FileCheck %s
// REQUIRES: stable-runtime

#include <pthread.h>

int *x;

void *AllocThread(void *arg) {
  x = new int;
  *x = 42;
  return NULL;
}

void *FreeThread(void *arg) {
  delete x;
  return NULL;
}

void *AccessThread(void *arg) {
  *x = 43;  // BOOM
  return NULL;
}

typedef void* (*callback_type)(void* arg);

void *RunnerThread(void *function) {
  pthread_t thread;
  pthread_create(&thread, NULL, (callback_type)function, NULL);
  pthread_join(thread, NULL);
  return NULL;
}

void RunThread(callback_type function) {
  pthread_t runner;
  pthread_create(&runner, NULL, RunnerThread, (void*)function);
  pthread_join(runner, NULL);
}

int main(int argc, char *argv[]) {
  RunThread(AllocThread);
  RunThread(FreeThread);
  RunThread(AccessThread);
  return (x != 0);
}

// CHECK: AddressSanitizer: heap-use-after-free
// CHECK: WRITE of size 4 at 0x{{.*}} thread T[[ACCESS_THREAD:[0-9]+]]
// CHECK: freed by thread T[[FREE_THREAD:[0-9]+]] here:
// CHECK: previously allocated by thread T[[ALLOC_THREAD:[0-9]+]] here:
// CHECK: Thread T[[ACCESS_THREAD]] created by T[[ACCESS_RUNNER:[0-9]+]] here:
// CHECK: Thread T[[ACCESS_RUNNER]] created by T0 here:
// CHECK: Thread T[[FREE_THREAD]] created by T[[FREE_RUNNER:[0-9]+]] here:
// CHECK: Thread T[[FREE_RUNNER]] created by T0 here:
// CHECK: Thread T[[ALLOC_THREAD]] created by T[[ALLOC_RUNNER:[0-9]+]] here:
// CHECK: Thread T[[ALLOC_RUNNER]] created by T0 here:
