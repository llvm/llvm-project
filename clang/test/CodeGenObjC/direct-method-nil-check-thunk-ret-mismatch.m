// RUN: %clang_cc1 -fobjc-emit-nil-check-thunk -fobjc-export-direct-methods -emit-llvm -fobjc-arc -triple arm64-apple-darwin %s -o - | FileCheck %s

__attribute__((objc_root_class))
@interface Root
- (Root *)method __attribute__((objc_direct));
@end

@implementation Root
// CHECK-LABEL: define internal ptr @"\01-[Root something]"(ptr noundef
- (id)something {
  // The compiler can reason that we can just call the method without nil check.
  // CHECK: %{{[^ ]*}} = call {{.*}} @"-<Root method>
  return [self method];
}
// The inner function should not contain any nil check anymore.
// CHECK-LABEL: define dso_local ptr @"-<Root method>_inner"(ptr noundef nonnull
// CHECK-NOT: br i1 %1, label %objc_direct_method.self_is_nil, label %objc_direct_method.cont

// The direct function contains the nil check.
// CHECK-LABEL: define dso_local ptr @"-<Root method>"(ptr noundef
// CHECK-LABEL: br i1 %1, label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
- (id)method {
  return self;
}
@end
