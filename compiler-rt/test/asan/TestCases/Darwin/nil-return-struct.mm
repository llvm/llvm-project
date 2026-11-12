// RUN: %clang_asan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

struct MyStruct {
  long a, b, c, d;
};

@interface MyClass: NSObject
- (MyStruct)methodWhichReturnsARect;
@end
@implementation MyClass
- (MyStruct)methodWhichReturnsARect {
  MyStruct s;
  s.a = 10;
  s.b = 20;
  s.c = 30;
  s.d = 40;
  return s;
}
@end

int main() {
  MyClass *myNil = nil;  // intentionally nil
  [myNil methodWhichReturnsARect];
  fprintf(stderr, "Hello world");
}

// CHECK-NOT: AddressSanitizer: stack-use-after-scope
// CHECK: Hello world
