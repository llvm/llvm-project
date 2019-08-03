// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <unistd.h>

int main() {
  pthread_mutex_t m;
  pthread_mutex_init(&m, 0);
  pthread_mutex_lock(&m);
  pthread_mutex_destroy(&m);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: destroy of a locked mutex
// CHECK:     #0 pthread_mutex_destroy
// CHECK:     #1 main
// CHECK:   and:
// CHECK:     #0 pthread_mutex_lock
// CHECK:     #1 main
// CHECK:   Mutex {{.*}} created at:
// CHECK:     #0 pthread_mutex_init
// CHECK:     #1 main
// CHECK: SUMMARY: ThreadSanitizer: destroy of a locked mutex{{.*}}main
