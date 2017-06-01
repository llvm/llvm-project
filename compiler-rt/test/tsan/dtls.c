// RUN: %clang_tsan %s -o %t
// RUN: %clang_tsan %s -DBUILD_SO -fPIC -o %t-so.so -shared
// RUN: %run %t 2>&1 | FileCheck %s

// Test that tsan cleans up dynamic TLS memory between reuse.

#include "test.h"

#ifndef BUILD_SO
#include <assert.h>
#include <dlfcn.h>

typedef volatile long *(* get_t)();
get_t GetTls;

void *Thread1(void *arg) {
  pthread_detach(pthread_self());
  volatile long *x = GetTls();
  *x = 42;
  fprintf(stderr, "stack: %p dtls: %p\n", &x, x);
  barrier_wait(&barrier);
  return 0;
}

void *Thread2(void *arg) {
  volatile long *x = GetTls();
  *x = 42;
  fprintf(stderr, "stack: %p dtls: %p\n", &x, x);
  return 0;
}

int main(int argc, char *argv[]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s-so.so", argv[0]);

  void *handle = dlopen(path, RTLD_LAZY);
  if (!handle) fprintf(stderr, "%s\n", dlerror());
  assert(handle != 0);
  GetTls = (get_t)dlsym(handle, "GetTls");
  assert(dlerror() == 0);

  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread1, 0);
  barrier_wait(&barrier);
  // Wait for actual thread termination without using pthread_join,
  // which would synchronize threads.
  sleep(1);
  pthread_create(&t[1], 0, Thread2, 0);
  pthread_join(t[1], 0);
  fprintf(stderr, "DONE\n");
  return 0;
}
#else  // BUILD_SO
__thread long huge_thread_local_array[1 << 17];
long *GetTls() {
  return &huge_thread_local_array[0];
}
#endif

// CHECK-NOT: ThreadSanitizer: data race
// CHECK: DONE
