// Check that Objective-C functions and methods are protected when stack probing is enabled
// RUN: %clang_cc1 -triple arm64-apple-darwin -emit-llvm -o- %s -fstack-check -Wno-objc-root-class | FileCheck %s

// Check that Objective-C functions and methods are not protected with stack probing by default
// RUN: %clang_cc1 -triple arm64-apple-darwin -emit-llvm -o- %s -Wno-objc-root-class | FileCheck %s --check-prefix=CHECK-DEFAULT

void f(void) {}

@interface C @end
@implementation C
+ (void)m1 {}
- (void)m2 {}
@end

// CHECK: define void @f() #[[ATTR_ID:[0-9]+]] {
// CHECK: define internal void @"\01+[C m1]"({{.*}}) #[[ATTR_ID]] {
// CHECK: define internal void @"\01-[C m2]"({{.*}}) #[[ATTR_ID]] {
// CHECK: attributes #[[ATTR_ID]] ={{.*}}"probe-stack"="__chkstk_darwin"
// CHECK-DEFAULT-NOT: "probe-stack"="__chkstk_darwin"
