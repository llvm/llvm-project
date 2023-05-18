// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include <stdio.h>

long global;

int main(int argc, const char *argv[]) {
  dispatch_semaphore_t done = dispatch_semaphore_create(0);

  dispatch_queue_t queue =
      dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);

  dispatch_source_t source =
      dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue);

  dispatch_source_set_timer(source, dispatch_walltime(NULL, 0), 1e9, 5);

  global = 42;

  dispatch_source_set_cancel_handler(source, ^{
    fprintf(stderr, "global = %ld\n", global);

    dispatch_semaphore_signal(done);
  });

  dispatch_resume(source);
  dispatch_cancel(source);

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);

  return 0;
}

// CHECK: global = 42
