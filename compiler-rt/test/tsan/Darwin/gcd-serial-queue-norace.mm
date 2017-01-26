// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

long global;

int main() {
  NSLog(@"Hello world.");
  NSLog(@"addr=%p\n", &global);

  dispatch_queue_t q1 = dispatch_queue_create("my.queue1", DISPATCH_QUEUE_CONCURRENT);
  dispatch_queue_t q2 = dispatch_queue_create("my.queue2", DISPATCH_QUEUE_SERIAL);

  global = 42;
  for (int i = 0; i < 10; i++) {
    dispatch_async(q1, ^{
      for (int i = 0; i < 100; i++) {
        dispatch_sync(q2, ^{
          global++;
        });
      }
    });
  }

  dispatch_barrier_async(q1, ^{
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
