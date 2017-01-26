// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

long global;

static const long nIter = 1000;

int main() {
  NSLog(@"Hello world.");

  global = 42;
  for (int i = 0; i < nIter; i++) {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
      dispatch_sync(dispatch_get_main_queue(), ^{
        global = i;

        if (i == nIter - 1) {
          CFRunLoopStop(CFRunLoopGetCurrent());
        }
      });
    });
  }

  CFRunLoopRun();
  NSLog(@"Done.");
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
