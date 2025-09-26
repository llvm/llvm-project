// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdio.h>

void *fn(void *) { return NULL; }

int main() {
  pthread_t th;
  int rc = pthread_create(&th, 0, fn, 0);
  if (rc)
    return rc;
  pthread_join(th, NULL);
  pthread_detach(th);
  return 0;
}

// CHECK: ThreadSanitizer: pthread_detach was called on thread 0 but it is dead.
