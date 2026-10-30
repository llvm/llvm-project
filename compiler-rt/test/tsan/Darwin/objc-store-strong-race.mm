// RUN: %clangxx_tsan %s -o %t-O0 -O0 -fobjc-arc -framework Foundation
// RUN: %deflake %run %t-O0 2>&1 | FileCheck %s
// RUN: %clangxx_tsan %s -o %t-O2 -O2 -fobjc-arc -framework Foundation
// RUN: %deflake %run %t-O2 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

@interface MyObj : NSObject {
@public
  __strong id strongField;
}
@end

@implementation MyObj
@end

int main() {
  barrier_init(&barrier, 2);

  MyObj *container = [MyObj new];
  id val1 = [NSObject new];
  id val2 = [NSObject new];

  dispatch_queue_t q =
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  dispatch_apply(2, q, ^(size_t i) {
    if (i == 0) {
      container->strongField = val1;
      barrier_wait(&barrier);
    } else {
      barrier_wait(&barrier);
      container->strongField = val2;
    }
  });

  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8 at {{.*}} by {{thread T.*|main thread}}:
// CHECK:     #0 {{.*}} objc-store-strong-race.mm:[[@LINE-10]]
// CHECK:   Previous write of size 8 at {{.*}} by {{thread T.*|main thread}}:
// CHECK:     #0 {{.*}} objc-store-strong-race.mm:[[@LINE-16]]
// CHECK: Done.
