// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// RUN: %clang_cc1 -fobjc-emit-nil-check-thunk -emit-llvm -fobjc-arc -triple arm64-apple-darwin %s -o - | FileCheck --check-prefixes=CHECK-THUNK %s

__attribute__((objc_root_class))
@interface Root
- (Root *)method __attribute__((objc_direct));
@end

@implementation Root
// CHECK-LABEL: define internal ptr @"\01-[Root something]"(
- (id)something {
  // CHECK: %{{[^ ]*}} = call {{.*}} @"\01-[Root method]"
  return [self method];
}

// CHECK-LABEL: define hidden ptr @"\01-[Root method]"(
- (id)method {
  return self;
}
@end
// The inner function should not contain any nil check anymore.
// CHECK-THUNK-LABEL: define hidden ptr @"\01-[Root method]_inner"(ptr noundef nonnull
// CHECK-THUNK-NOT: br i1 %1, label %objc_direct_method.self_is_nil, label %objc_direct_method.cont

// The direct function contains the nil check.
// CHECK-THUNK-LABEL: define hidden ptr @"\01-[Root method]"(ptr noundef
// CHECK-THUNK-LABEL: br i1 %1, label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
