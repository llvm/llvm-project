// RUN: %clang_tsan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <os/lock.h>
#include <pthread.h>
#include <stdio.h>

long global_variable;
os_unfair_lock lock1 = OS_UNFAIR_LOCK_INIT, lock2 = OS_UNFAIR_LOCK_INIT;

void *LockInAscendingOrder(void *a) {
  os_unfair_lock_lock(&lock1);
  os_unfair_lock_lock(&lock2);
  global_variable++;
  os_unfair_lock_unlock(&lock2);
  os_unfair_lock_unlock(&lock1);
  return NULL;
}

void *LockInDescendingOrder(void *a) {
  os_unfair_lock_lock(&lock2);
  os_unfair_lock_lock(&lock1);
  global_variable++;
  os_unfair_lock_unlock(&lock1);
  os_unfair_lock_unlock(&lock2);
  return NULL;
}

int main() {
  pthread_t t1, t2;
  global_variable = 0;
  pthread_create(&t1, NULL, LockInAscendingOrder, NULL);
  pthread_join(t1, NULL);
  pthread_create(&t2, NULL, LockInDescendingOrder, NULL);
  pthread_join(t2, NULL);
  // CHECK: WARNING: ThreadSanitizer: lock-order-inversion (potential deadlock)
  fprintf(stderr, "global_variable = %ld\n", global_variable);
  // CHECK: global_variable = 2
}
