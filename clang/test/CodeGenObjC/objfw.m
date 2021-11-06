// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fobjc-runtime=objfw -emit-llvm -o - %s | FileCheck %s

// Test the ObjFW runtime.

@interface Test0
+ (void) test;
@end
void test0(void) {
  [Test0 test];
}
// CHECK-LABEL:    define{{.*}} void @test0()
// CHECK:      [[T0:%.*]] = call i8* (i8*, i8*, ...)* @objc_msg_lookup(i8* bitcast (i64* @_OBJC_CLASS_Test0 to i8*),
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* (i8*, i8*, ...)* [[T0]] to void (i8*, i8*)*
// CHECK-NEXT: call void [[T1]](i8* bitcast (i64* @_OBJC_CLASS_Test0 to i8*), 
// CHECK-NEXT: ret void
