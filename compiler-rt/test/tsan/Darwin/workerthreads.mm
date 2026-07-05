// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

long global;

int main() {
  fprintf(stderr, "Hello world.\n");
  print_address("addr=", 1, &global);
  barrier_init(&barrier, 2);

  global = 42;
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    global = 43;
    barrier_wait(&barrier);
  });

  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    barrier_wait(&barrier);
    global = 44;

    dispatch_sync(dispatch_get_main_queue(), ^{
      CFRunLoopStop(CFRunLoopGetCurrent());
    });
  });

  CFRunLoopRun();
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Write of size 8
// CHECK: Previous write of size 8
// CHECK: Location is global
// CHECK: Thread {{.*}} is a GCD worker thread
// CHECK-NOT: failed to restore the stack
// CHECK: Thread {{.*}} is a GCD worker thread
// CHECK-NOT: failed to restore the stack
// CHECK: Done.
