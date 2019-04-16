// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include "dispatch/dispatch.h"

#include <stdio.h>

long global;

int main() {
  fprintf(stderr, "Hello world.\n");
  dispatch_semaphore_t done = dispatch_semaphore_create(0);

  dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  global = 42;

  dispatch_group_t g = dispatch_group_create();
  dispatch_group_async(g, q, ^{
    global = 43;
  });
  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);

  global = 44;

  dispatch_group_enter(g);
  dispatch_async(q, ^{
    global = 45;
    dispatch_group_leave(g);
  });
  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);

  global = 46;

  dispatch_group_enter(g);
  dispatch_async(q, ^{
    global = 47;
    dispatch_group_leave(g);
  });
  dispatch_group_notify(g, q, ^{
    global = 48;

    dispatch_semaphore_signal(done);
  });

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
