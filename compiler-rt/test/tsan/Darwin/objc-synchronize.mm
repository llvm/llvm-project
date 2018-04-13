// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc %darwin_min_target_with_full_runtime_arc_support
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

@interface MyClass : NSObject {
  long field;
}
@property (nonatomic, readonly) long value;
@end

dispatch_group_t group;

@implementation MyClass

- (void) start {
  dispatch_queue_t q = dispatch_queue_create(NULL, NULL);
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    for (int i = 0; i < 1000; i++) {
      dispatch_async(q, ^{
        @synchronized(self) {
          self->field = i;
        }
      });
    }
  });
}

- (long) value {
  @synchronized(self) {
    return self->field;
  }
}

- (void)dealloc {
  dispatch_group_leave(group);
}

@end

int main() {
  group = dispatch_group_create();
  @autoreleasepool {
    for (int j = 0; j < 100; ++j) {
      dispatch_group_enter(group);
      MyClass *obj = [[MyClass alloc] init];
      [obj start];
      long x = obj.value;
      (void)x;
    }
  }
  dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
  NSLog(@"Hello world");
}

// CHECK: Hello world
// CHECK-NOT: WARNING: ThreadSanitizer
