// Test that threads are reused.
// On Android, pthread_* are in libc.so. So the `-lpthread` is not supported.
// Use `-pthread` so that its driver will DTRT (ie., ignore it).
// RUN: %clangxx_lsan %s -o %t -pthread && %run %t

#include <assert.h>
#include <dirent.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

// Number of threads to create. This value is greater than kMaxThreads in
// lsan_thread.cpp so that we can test that thread contexts are not being
// reused.
static const size_t kTestThreads = 10000;

// Limit the number of simultaneous threads to avoid reaching the limit.
static const size_t kTestThreadsBatch = 100;

void *null_func(void *args) {
  return NULL;
}

int count_threads() {
  DIR *d = opendir("/proc/self/task");
  assert(d);
  int count = 0;
  while (readdir(d))
    ++count;
  closedir(d);
  assert(count);
  return count;
}

int main(void) {
  for (size_t i = 0; i < kTestThreads; i += kTestThreadsBatch) {
    for (size_t j = 0; j < kTestThreadsBatch; ++j) {
      pthread_t thread;
      assert(pthread_create(&thread, NULL, null_func, NULL) == 0);
      pthread_detach(thread);
    }
    while (count_threads() > 10)
      sched_yield();
  }
  return 0;
}
