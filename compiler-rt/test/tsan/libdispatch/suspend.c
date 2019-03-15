// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <dispatch/dispatch.h>

#include <stdio.h>

long my_global = 0;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  dispatch_queue_t q1 = dispatch_queue_create("queue1", NULL);
  dispatch_queue_t q2 = dispatch_queue_create("queue2", NULL);
  dispatch_group_t g = dispatch_group_create();

  dispatch_sync(q1, ^{
    dispatch_suspend(q1);
    dispatch_async(q2, ^{
      my_global++;
      dispatch_resume(q1);
    });
  });

  dispatch_sync(q1, ^{
    my_global++;
  });

  dispatch_sync(q1, ^{
    dispatch_suspend(q1);
    dispatch_group_enter(g);
    dispatch_async(q1,^{ my_global++; });
    dispatch_async(q1,^{ my_global++; });
    dispatch_async(q1,^{ my_global++; dispatch_group_leave(g); });
    my_global++;
    dispatch_resume(q1);
  });

  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);

  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: Hello world.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
