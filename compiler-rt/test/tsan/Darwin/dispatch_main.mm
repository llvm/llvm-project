// Check that we don't crash when dispatch_main calls pthread_exit which
// quits the main thread.

// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

int main() {
  NSLog(@"Hello world");

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);

  dispatch_async(q, ^{
    NSLog(@"1");
  });

  dispatch_async(q, ^{
    NSLog(@"2");
  });

  dispatch_async(q, ^{
    NSLog(@"3");

    dispatch_async(dispatch_get_main_queue(), ^{
      NSLog(@"Done.");
      sleep(1);
      exit(0);
    });
  });

  dispatch_main();
}

// CHECK: Hello world
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK-NOT: CHECK failed
