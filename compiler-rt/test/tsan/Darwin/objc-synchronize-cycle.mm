// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc
// RUN:                                   not %run %t 2>&1 | FileCheck %s
// RUN: %env_tsan_opts=detect_deadlocks=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_tsan_opts=detect_deadlocks=0     %run %t 2>&1 | FileCheck %s --check-prefix=DISABLED

#import <Foundation/Foundation.h>

int main() {
  @autoreleasepool {
    NSObject* obj1 = [NSObject new];
    NSObject* obj2 = [NSObject new];

    // obj1 -> obj2
    @synchronized(obj1) {
      @synchronized(obj2) {
      }
    }

    // obj1 -> obj1
    @synchronized(obj2) {
      @synchronized(obj1) {
// CHECK: ThreadSanitizer: lock-order-inversion (potential deadlock)
      }
    }
  }

  NSLog(@"PASS");
// DISABLED-NOT: ThreadSanitizer
// DISABLED: PASS
  return 0;
}
