// Check that sanitizer does not leak args when passing to a child thread.

// RUN: %clangxx_lsan -pthread %s -o %t -DLEAK_ARG && %run not %t 10 2>&1 | FileCheck %s --check-prefixes=LEAK,LEAK123
// RUN: %clangxx_lsan -pthread %s -o %t -DLEAK_RES && %run not %t 10 2>&1 | FileCheck %s --check-prefixes=LEAK,LEAK234
// RUN: %clangxx_lsan -pthread %s -o %t -DLEAK_DETACH && %run not %t 10 2>&1 | FileCheck %s --check-prefixes=LEAK,LEAK234

// FIXME: Remove "not". There is no leak.
// False LEAK123 is broken for HWASAN.
// False LEAK234 is broken for ASAN, HWASAN, LSAN.
// RUN: %clangxx_lsan -pthread %s -o %t && %run not %t 10

#include <pthread.h>
#include <stdlib.h>

#include <sanitizer/lsan_interface.h>

static void *thread_free(void *args) {
#ifndef LEAK_ARG
  free(args);
#endif
  return malloc(234);
}

int main(int argc, char **argv) {
  int n = atoi(argv[1]);
  for (int i = 0; i < n; ++i) {
    pthread_t threads[10];

    for (auto &thread : threads) {
      pthread_create(&thread, 0, thread_free, malloc(123));
      if (__lsan_do_recoverable_leak_check())
        return 1;
    }

    for (auto &thread : threads) {
#ifdef LEAK_DETACH
      pthread_detach(thread);
      continue;
#endif
      void *retval = 0;
      pthread_join(thread, &retval);
#ifndef LEAK_RES
      free(retval);
#endif
    }
  }
  return 0;
}

// LEAK: LeakSanitizer: detected memory leaks
// LEAK123: in main
// LEAK234: in thread_free
