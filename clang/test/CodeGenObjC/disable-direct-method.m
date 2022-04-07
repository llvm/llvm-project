// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple x86_64-apple-darwin10 -fobjc-disable-direct-methods-for-testing %s -o - | FileCheck %s

@interface Y
@property (direct) int direct_property;
@end
@implementation Y @end

// CHECK: @OBJC_PROP_NAME_ATTR_ = private unnamed_addr constant [16 x i8] c"direct_property\00"
// CHECK: @"_OBJC_$_PROP_LIST_Y" =
// CHECK-SAME: @OBJC_PROP_NAME_ATTR_,

@interface X
-(void)m __attribute__((objc_direct));
@end

// CHECK-LABEL: define void @f
void f(X *x) {
  [x m];

  // CHECK: call void bitcast ({{.*}} @objc_msgSend to {{.*}})
}
