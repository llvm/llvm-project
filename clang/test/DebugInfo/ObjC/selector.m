// RUN: %clang_cc1 -emit-llvm  -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: objc_selector
@interface MyClass {
}
- (id)init;
@end

@implementation MyClass
- (id) init
{
  return self;
}
@end
