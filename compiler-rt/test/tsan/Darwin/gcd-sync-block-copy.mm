// This test verifies that dispatch_sync() doesn't actually copy the block under TSan (without TSan, it doesn't).

// RUN: %clang_tsan -fno-sanitize=thread %s -o %t_no_tsan -framework Foundation
// RUN: %run %t_no_tsan 2>&1 | FileCheck %s

// RUN: %clang_tsan %s -o %t_with_tsan -framework Foundation
// RUN: %run %t_with_tsan 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

@interface MyClass : NSObject
@end

@implementation MyClass
- (instancetype)retain {
  // Copying the dispatch_sync'd block below will increment the retain count of
  // this object. Abort if that happens.
  abort();
}
@end

int main(int argc, const char* argv[]) {
  dispatch_queue_t q = dispatch_queue_create("my.queue", NULL);
  id object = [[MyClass alloc] init];
  dispatch_sync(q, ^{
    NSLog(@"%@", object);
  });
  [object release];
  NSLog(@"Done.");
  return 0;
}

// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
