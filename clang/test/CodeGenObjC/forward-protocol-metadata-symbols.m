// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-10.14 -emit-llvm -x objective-c %s -o - | FileCheck %s
// rdar://16203115

@interface NSObject @end

@protocol P0 @end

@interface A : NSObject <P0>
+(Class) getClass;
@end

@implementation A
+(Class) getClass { return self; }
@end

int main(void) {
  Protocol *P0 = @protocol(P0);
  return 0;
}

// CHECK: @"_OBJC_PROTOCOL_$_P0" = weak hidden global
// CHECK: @"_OBJC_LABEL_PROTOCOL_$_P0" = weak hidden global
// CHECK: @"_OBJC_CLASS_PROTOCOLS_$_A" = internal global
// CHECK: @"_OBJC_PROTOCOL_REFERENCE_$_P0" = weak hidden global

// CHECK: llvm.used = appending global [3 x i8*]
// CHECK-SAME: "_OBJC_PROTOCOL_$_P0"
// CHECK-SAME: "_OBJC_LABEL_PROTOCOL_$_P0"
// CHECK-SAME: "_OBJC_PROTOCOL_REFERENCE_$_P0"

// CHECK: llvm.compiler.used = appending global [7 x i8*]
// CHECK-SAME: OBJC_CLASS_NAME_
// CHECK-SAME: OBJC_METH_VAR_NAME_
// CHECK-SAME: OBJC_METH_VAR_TYPE_
// CHECK-SAME: "_OBJC_$_CLASS_METHODS_A"
// CHECK-SAME: OBJC_CLASS_NAME_.1
// CHECK-SAME: "_OBJC_CLASS_PROTOCOLS_$_A"
// CHECK-SAME: "OBJC_LABEL_CLASS_$"
// CHECK-SAME: section "llvm.metadata"
