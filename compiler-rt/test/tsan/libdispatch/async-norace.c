// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include <stdio.h>

long global;

int main() {
  fprintf(stderr, "Hello world.\n");
  dispatch_semaphore_t done = dispatch_semaphore_create(0);

  global = 42;
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    global = 43;

    dispatch_semaphore_signal(done);
  });

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
