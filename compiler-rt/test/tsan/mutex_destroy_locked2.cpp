// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <unistd.h>

void *thread(void *arg) {
  pthread_mutex_t m;
  pthread_mutex_init(&m, 0);
  pthread_mutex_lock(&m);
  pthread_mutex_destroy(&m);
  return 0;
}

int main() {
  pthread_t th;
  pthread_create(&th, 0, thread, 0);
  pthread_join(th, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: destroy of a locked mutex
// CHECK:     #0 pthread_mutex_destroy
// CHECK:     #1 thread
// CHECK:   and:
// CHECK:     #0 pthread_mutex_lock
// CHECK:     #1 thread
// CHECK:   Mutex {{.*}} created at:
// CHECK:     #0 pthread_mutex_init
// CHECK:     #1 thread
// CHECK: SUMMARY: ThreadSanitizer: destroy of a locked mutex {{.*}} in thread
