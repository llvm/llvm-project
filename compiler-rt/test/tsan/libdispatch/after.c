// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include <stdio.h>

long my_global;
long my_global2;
dispatch_semaphore_t done;

void callback(void *context) {
  my_global2 = 42;

  dispatch_semaphore_signal(done);
}

int main(int argc, const char *argv[]) {
  fprintf(stderr, "start\n");
  done = dispatch_semaphore_create(0);

  dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  my_global = 10;
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_MSEC)), q, ^{
    my_global = 42;

    dispatch_semaphore_signal(done);
  });

  my_global2 = 10;
  dispatch_after_f(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_MSEC)), q, NULL, &callback);

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK: done
