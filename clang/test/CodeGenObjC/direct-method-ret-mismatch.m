// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

__attribute__((objc_root_class))
@interface Root
- (Root *)method __attribute__((objc_direct));
@end

@implementation Root
// CHECK-LABEL: define internal i8* @"\01-[Root something]"(
- (id)something {
  // CHECK: %{{[^ ]*}} = call {{.*}} @"\01-[Root method]"
  return [self method];
}

// CHECK-LABEL: define hidden i8* @"\01-[Root method]"(
- (id)method {
  return self;
}
@end
