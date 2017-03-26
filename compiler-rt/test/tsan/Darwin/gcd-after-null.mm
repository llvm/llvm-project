// Regression test to make sure we don't crash when dispatch_after is called with a NULL queue.

// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

int main(int argc, const char *argv[]) {
  fprintf(stderr, "start\n");

  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_MSEC)), NULL, ^{
    dispatch_async(dispatch_get_main_queue(), ^{
      CFRunLoopStop(CFRunLoopGetMain());
    });
  });
  CFRunLoopRun();

  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK: done
