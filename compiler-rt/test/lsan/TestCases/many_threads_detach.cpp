// Test that threads are reused.
// RUN: %clangxx_lsan %s -o %t -lpthread && %run %t

#include <pthread.h>
#include <stdlib.h>

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
    pthread_create(&thread, NULL, null_func, NULL);
    pthread_detach(thread);
  }
  return 0;
}
