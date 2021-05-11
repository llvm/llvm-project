// Test that threads are reused.
// On Android, pthread_* are in libc.so. So the `-lpthread` is not supported.
// Use `-pthread` so that its driver will DTRT (ie., ignore it).
// RUN: %clangxx_lsan %s -o %t -pthread && %run %t

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

// Number of threads to create. This value is greater than kMaxThreads in
// lsan_thread.cpp so that we can test that thread contexts are not being
// reused.
static const size_t kTestThreads = 10000;

void *null_func(void *args) {
  return NULL;
}

int main(void) {
  for (size_t i = 0; i < kTestThreads; i++) {
    pthread_t thread;
    if (pthread_create(&thread, NULL, null_func, NULL) == 0)
      pthread_detach(thread);
  }
  return 0;
}
