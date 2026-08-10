// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

typedef int32_t OSSpinLock;
extern "C" void OSSpinLockLock(OSSpinLock *);
extern "C" void OSSpinLockUnlock(OSSpinLock *);

int Global;
OSSpinLock lock;

void *Thread(void *x) {
  OSSpinLockLock(&lock);
  Global++;
  OSSpinLockUnlock(&lock);
  return NULL;
}

int main() {
  fprintf(stderr, "Hello world.\n");

  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread, NULL);
  pthread_create(&t[1], NULL, Thread, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
