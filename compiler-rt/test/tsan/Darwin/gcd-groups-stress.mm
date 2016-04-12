// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

void notify_callback(void *context) {
  // Do nothing.
}

int main() {
  NSLog(@"Hello world.");

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

  NSLog(@"Done.");
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK-NOT: CHECK failed
