// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %env_tsan_opts=detect_deadlocks=1 %run %t 2>&1 | FileCheck %s
// RUN: %env_tsan_opts=detect_deadlocks=0 %run %t 2>&1 | FileCheck %s

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
      }
    }
  }

  NSLog(@"PASS");
// CHECK-NOT: ThreadSanitizer
// CHECK: PASS
// CHECK-NOT: ThreadSanitizer
  return 0;
}
