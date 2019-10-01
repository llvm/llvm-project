// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc %darwin_min_target_with_full_runtime_arc_support
// RUN:     %run %t 6 2>&1 | FileCheck %s --check-prefix=SIX
// RUN: not %run %t 7 2>&1 | FileCheck %s --check-prefix=SEVEN

#import <Foundation/Foundation.h>

static bool isTaggedPtr(id obj) {
  uintptr_t ptr = (uintptr_t) obj;
  return (ptr & 0x8000000000000001ull) != 0;
}

int main(int argc, char* argv[]) {
  assert(argc == 2);
  int arg = atoi(argv[1]);

  @autoreleasepool {
    NSObject* obj = [NSObject new];
    NSObject* num1 = @7;
    NSObject* num2 = [NSNumber numberWithInt:arg];

    assert(!isTaggedPtr(obj));
    assert(isTaggedPtr(num1) && isTaggedPtr(num2));

    // obj -> num1 (includes num2)
    @synchronized(obj) {
      @synchronized(num1) {
      }
    }

    // num2 -> obj1
    @synchronized(num2) {
      @synchronized(obj) {
// SEVEN: ThreadSanitizer: lock-order-inversion (potential deadlock)
      }
    }
  }

  NSLog(@"PASS");
// SIX-NOT: ThreadSanitizer
// SIX: PASS
  return 0;
}
