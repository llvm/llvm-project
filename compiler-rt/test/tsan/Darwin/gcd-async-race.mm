// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %deflake %run %t 2>&1

#import <Foundation/Foundation.h>

#import "../test.h"

long global;

int main() {
  NSLog(@"Hello world.");
  NSLog(@"addr=%p\n", &global);
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
  NSLog(@"Done.");
}

// CHECK: Hello world.
// CHECK: addr=[[ADDR:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'global' at [[ADDR]] (global_race.cc.exe+0x{{[0-9,a-f]+}})
// CHECK: Done.
