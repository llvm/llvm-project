// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <Availability.h>
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

#if defined(__MAC_15_0)
void *ThreadWithFlags(void *a) {
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wunguarded-availability-new"
  os_unfair_lock_lock_with_flags(&lock, OS_UNFAIR_LOCK_FLAG_ADAPTIVE_SPIN);
#  pragma clang diagnostic pop
  global_variable++;
  os_unfair_lock_unlock(&lock);
  return NULL;
}
#endif

int main() {
  pthread_t t1, t2;
  global_variable = 0;
  pthread_create(&t1, NULL, Thread, NULL);
  pthread_create(&t2, NULL, Thread, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  fprintf(stderr, "global_variable = %ld\n", global_variable);

  // CHECK: global_variable = 2

  void *(*func)(void *) = Thread;
  char flags_available = 0;
#if defined(__MAC_15_0)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wunguarded-availability-new"
  if (&os_unfair_lock_lock_with_flags) {
#  pragma clang diagnostic pop
    func = ThreadWithFlags;
    flags_available = 1;
  }
#endif

  pthread_create(&t1, NULL, func, NULL);
  pthread_create(&t2, NULL, func, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  fprintf(stderr,
          "global_variable = %ld, os_unfair_lock_lock_with_flags %savailable\n",
          global_variable, flags_available ? "" : "un");
}

// CHECK: global_variable = 4
