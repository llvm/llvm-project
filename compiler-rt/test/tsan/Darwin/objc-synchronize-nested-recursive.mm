// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc %darwin_min_target_with_full_runtime_arc_support
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

int main() {
  @autoreleasepool {
    NSObject* obj1 = [NSObject new];
    NSObject* obj2 = [NSObject new];

    @synchronized(obj1) {
      @synchronized(obj1) {
        NSLog(@"nested 1-1");
// CHECK: nested 1-1
      }
    }

    @synchronized(obj1) {
      @synchronized(obj2) {
        @synchronized(obj1) {
          @synchronized(obj2) {
            NSLog(@"nested 1-2-1-2");
// CHECK: nested 1-2-1-2
          }
        }
      }
    }

  }

  NSLog(@"PASS");
// CHECK-NOT: ThreadSanitizer
// CHECK: PASS
  return 0;
}
