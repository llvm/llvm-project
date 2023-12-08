// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <dispatch/dispatch.h>

#include <stdio.h>

void notify_callback(void *context) {
  // Do nothing.
}

int main() {
  fprintf(stderr, "Hello world.");

  dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  for (int i = 0; i < 300000; i++) {
    dispatch_group_t g = dispatch_group_create();
    dispatch_group_enter(g);
    dispatch_async(q, ^{
      dispatch_group_leave(g);
    });
    dispatch_group_notify(g, q, ^{
      // Do nothing.
    });
    dispatch_release(g);
  }

  for (int i = 0; i < 300000; i++) {
    dispatch_group_t g = dispatch_group_create();
    dispatch_group_enter(g);
    dispatch_async(q, ^{
      dispatch_group_leave(g);
    });
    dispatch_group_notify_f(g, q, NULL, &notify_callback);
    dispatch_release(g);
  }

  fprintf(stderr, "Done.");
}

// CHECK: Hello world.
// CHECK: Done.
