// RUN: %clangxx -pthread %s -o %t

// RUN: %env_tool_opts=detect_invalid_join=true not %run %t 0 2>&1 | FileCheck %s
// RUN: %env_tool_opts=detect_invalid_join=true not %run %t 1 2>&1 | FileCheck %s
// RUN: %env_tool_opts=detect_invalid_join=true not %run %t 2 2>&1 | FileCheck %s
// RUN: %env_tool_opts=detect_invalid_join=true not %run %t 3 2>&1 | FileCheck %s --check-prefix=DETACH

// REQUIRES: glibc && (asan || hwasan || lsan)

#include <assert.h>
#include <ctime>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

static void *fn(void *args) {
  sleep(1);
  return nullptr;
}

int main(int argc, char **argv) {
  int n = atoi(argv[1]);
  pthread_t thread;
  assert(!pthread_create(&thread, nullptr, fn, nullptr));
  void *res;
  if (n == 0) {
    while (pthread_tryjoin_np(thread, &res))
      sleep(1);
    pthread_tryjoin_np(thread, &res);
  } else if (n == 1) {
    timespec tm = {0, 1};
    while (pthread_timedjoin_np(thread, &res, &tm))
      sleep(1);
    pthread_timedjoin_np(thread, &res, &tm);
  } else if (n == 2) {
    assert(!pthread_join(thread, &res));
    pthread_join(thread, &res);
  } else if (n == 3) {
    assert(!pthread_detach(thread));
    pthread_join(thread, &res);
  }
  // CHECK: Joining already joined thread
  // DETACH: Joining detached thread
  return 0;
}
