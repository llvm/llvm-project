// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

long global;

int main(int argc, const char *argv[]) {
  dispatch_queue_t target_queue = dispatch_queue_create(NULL, DISPATCH_QUEUE_SERIAL);
  dispatch_queue_t q1 = dispatch_queue_create(NULL, DISPATCH_QUEUE_CONCURRENT);
  dispatch_queue_t q2 = dispatch_queue_create(NULL, DISPATCH_QUEUE_CONCURRENT);
  dispatch_set_target_queue(q1, target_queue);
  dispatch_set_target_queue(q2, target_queue);

  for (int i = 0; i < 100000; i++) {
    dispatch_async(q1, ^{
      global++;

      if (global == 200000) {
        dispatch_sync(dispatch_get_main_queue(), ^{
          CFRunLoopStop(CFRunLoopGetCurrent());
        });
      }
    });
    dispatch_async(q2, ^{
      global++;

      if (global == 200000) {
        dispatch_sync(dispatch_get_main_queue(), ^{
          CFRunLoopStop(CFRunLoopGetCurrent());
        });
      }
    });
  }

  CFRunLoopRun();
  NSLog(@"Done.");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer
