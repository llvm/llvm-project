// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s

@protocol X
@optional
- (id)x;
@required
+ (void*)y;
@property int reqProp;
@optional
@property int optProp;
@end

// Check that we get some plausible-looking method lists.
// CHECK: internal global { i32, i32, [2 x { i8*, i8* }] } { i32 2, i32 16, 
// CHECK-SAME: @".objc_selector_reqProp_i16\010:8"
// CHECK-SAME: @".objc_selector_setReqProp:_v20\010:8i16"
// CHECK: internal global { i32, i32, [3 x { i8*, i8* }] } { i32 3, i32 16,
// CHECK-SAME: @".objc_selector_x_\0116\010:8"
// CHECK-SAME: @".objc_selector_optProp_i16\010:8"
// CHECK-SAME: @".objc_selector_setOptProp:_v20\010:8i16"


// Check that we're emitting the protocol and a correctly initialised
// indirection variable.
// CHECK: @._OBJC_PROTOCOL_X ={{.*}} global
// CHECK-SAME: , section "__objc_protocols", comdat, align 8
// CHECK: @._OBJC_REF_PROTOCOL_X = linkonce_odr global
// CHECK-SAME: @._OBJC_PROTOCOL_X
// CHECK-SAME: , section "__objc_protocol_refs", comdat, align 8


// Check that we load from the indirection variable on protocol references.
// CHECK: define{{.*}} i8* @x()
// CHECK:   = load 
// CHECK-SAME: @._OBJC_REF_PROTOCOL_X, align 8
void *x(void)
{
	return @protocol(X);
}
