// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

NSString *tagged_string = nil;

@interface MyClass : NSObject {
  long field;
}
@property(nonatomic, readonly) long value;
@end

dispatch_group_t group;

@implementation MyClass

- (void)start {
  dispatch_queue_t q = dispatch_queue_create(NULL, NULL);
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    for (int i = 0; i < 10; i++) {
      dispatch_async(q, ^{
        @synchronized(tagged_string) {
          self->field = i;
        }
      });
    }
  });
}

- (long)value {
  @synchronized(tagged_string) {
    return self->field;
  }
}

- (void)dealloc {
  dispatch_group_leave(group);
}

@end

int main() {
  tagged_string = [NSString stringWithFormat:@"%s", "abc"];
  uintptr_t tagged_string_bits = (uintptr_t)tagged_string;
  assert((tagged_string_bits & 0x8000000000000001ull) != 0);
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
