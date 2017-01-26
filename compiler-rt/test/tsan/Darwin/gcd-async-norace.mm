// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

long global;

int main() {
  NSLog(@"Hello world.");

  global = 42;
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    global = 43;

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
