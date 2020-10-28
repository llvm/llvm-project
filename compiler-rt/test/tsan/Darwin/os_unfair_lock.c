// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <os/lock.h>
#include <pthread.h>
#include <stdio.h>

long global_variable;
os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;

void *Thread(void *a) {
  os_unfair_lock_lock(&lock);
  global_variable++;
  os_unfair_lock_unlock(&lock);
  return NULL;
}

int main() {
  pthread_t t1, t2;
  global_variable = 0;
  pthread_create(&t1, NULL, Thread, NULL);
  pthread_create(&t2, NULL, Thread, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  fprintf(stderr, "global_variable = %ld\n", global_variable);
}

// CHECK: global_variable = 2
