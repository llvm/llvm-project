// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -emit-llvm -fobjc-runtime=gnustep-2.2 -o - %s | FileCheck %s

@interface X
@end

@implementation X
//- (int)x __attribute__((objc_direct)) { return 12; }
- (int)x __attribute__((objc_direct)) { return 12; }

// Check that the name is mangled like Objective-C methods and contains a nil check
// CHECK-LABEL: @_i_X__x
// CHECK: icmp eq ptr %0, null

+ (int)clsMeth __attribute__((objc_direct)) { return 42; }
// Check that the name is mangled like Objective-C methods and contains an initialisation check
// CHECK-LABEL: @_c_X__clsMeth
// CHECK: getelementptr inbounds { ptr, ptr, ptr, i64, i64 }, ptr %0, i32 0, i32 4
// CHECK: load i64, ptr %1, align 64
// CHECK: and i64 %2, 256
// CHECK: objc_direct_method.class_uninitialized:
// CHECK: call void @objc_send_initialize(ptr %0)

@end

// Check that the call sides are set up correctly.
void callCls(void)
{
	// CHECK: call i32 @_c_X__clsMeth
	[X clsMeth];
}

void callInstance(X *x)
{
	// CHECK: call i32 @_i_X__x
	[x x];
}

