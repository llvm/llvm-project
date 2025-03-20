// Test handling of arg and retval of child thread.

// RUN: %clangxx_lsan -pthread %s -o %t
// RUN: not %run %t 10 1 0 0 2>&1 | FileCheck %s --check-prefixes=LEAK,LEAK123
// RUN: not %run %t 10 0 1 0 2>&1 | FileCheck %s --check-prefixes=LEAK,LEAK234
// RUN: not %run %t 10 0 0 1 2>&1 | FileCheck %s --check-prefixes=LEAK,LEAK234
// RUN: %run %t 10 0 0 0

// This test appears to be flaky on x86_64-darwin buildbots.
// UNSUPPORTED: darwin

#include <pthread.h>
#include <stdlib.h>

#include <sanitizer/lsan_interface.h>

int detach;
int leak_arg;
int leak_retval;

static void *thread_free(void *args) {
  if (!leak_arg)
    free(args);
  return malloc(234);
}

int main(int argc, char **argv) {
  int n = atoi(argv[1]);
  leak_arg = atoi(argv[2]);
  leak_retval = atoi(argv[3]);
  detach = atoi(argv[4]);
  for (int i = 0; i < n; ++i) {
    pthread_t threads[10];

    for (auto &thread : threads) {
      pthread_create(&thread, 0, thread_free, malloc(123));
      if (__lsan_do_recoverable_leak_check())
        return 1;
    }

    for (auto &thread : threads) {
      if (detach) {
        pthread_detach(thread);
        continue;
      }
      void *retval = 0;
      pthread_join(thread, &retval);
      if (!leak_retval)
        free(retval);
    }
  }
  return 0;
}

// LEAK: LeakSanitizer: detected memory leaks
// LEAK123: in main
// LEAK234: in thread_free
