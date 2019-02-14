// RUN: %clangxx_msan %s -o %t && %run %t 2>&1 | FileCheck %s

#include <atomic>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" int
__cxa_atexit(void (*func)(void *), void *arg, void *d);

void handler(void *) {
}

std::atomic_int counter;

void *thread(void *) {
  for (int i = 0; i < 10000; ++i) {
    __cxa_atexit(&handler, 0, (void *)&handler);
    ++counter;
  }
  return 0;
}

int main(void) {
  printf("TEST_MAIN\n");
  pthread_t pt;
  for (int i = 0; i < 2; ++i)
    pthread_create(&pt, 0, &thread, 0);
  while (counter < 1000) {
  };
  return 0;
}
// CHECK: TEST_MAIN
// CHECK-NOT: MemorySanitizer
