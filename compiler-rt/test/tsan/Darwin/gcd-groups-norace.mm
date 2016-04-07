// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

long global;

int main() {
  NSLog(@"Hello world.");
  NSLog(@"addr=%p\n", &global);

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

    dispatch_sync(dispatch_get_main_queue(), ^{
      CFRunLoopStop(CFRunLoopGetCurrent());
    });
  });

  CFRunLoopRun();
  NSLog(@"Done.");
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
