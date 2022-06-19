// Check that dispatch_once() is always intercepted.

// RUN: %clang_tsan %s -o %t
// RUN: not %env_tsan_opts=ignore_noninstrumented_modules=0 %run %t 2>&1 | FileCheck %s

#include <dispatch/dispatch.h>

#include <pthread.h>
#include <stdio.h>

#include "../test.h"

long g = 0;
long h = 0;

__attribute__((disable_sanitizer_instrumentation))
void f() {
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    g++;
  });
  h++;
}

__attribute__((disable_sanitizer_instrumentation))
void __tsan_on_report() {
  fprintf(stderr, "Report.\n");

  // Without these annotations this test deadlocks for COMPILER_RT_DEBUG=ON
  // builds.  Conceptually, the TSan runtime does not support reentrancy from
  // runtime callbacks, but the main goal here is just to check that
  // dispatch_once() is always intercepted.
  AnnotateIgnoreSyncBegin(__FILE__, __LINE__);
  f();
  AnnotateIgnoreSyncEnd(__FILE__, __LINE__);
}

int main() {
  fprintf(stderr, "Hello world.\n");

  f();

  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_unlock(&mutex); // Unlock of an unlocked mutex

  fprintf(stderr, "g = %ld.\n", g);
  fprintf(stderr, "h = %ld.\n", h);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Report.
// CHECK: g = 1
// CHECK: h = 2
// CHECK: Done.
