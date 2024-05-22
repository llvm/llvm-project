// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fdebugger-support -funknown-anytype -emit-llvm -o - %s | FileCheck %s

@interface A @end
void test0(A *a) {
  (void) [a test0: (float) 2.0];
}
// CHECK-LABEL: define{{.*}} void @_Z5test0P1A(
// CHECK: call void @objc_msgSend(

@interface B
- (void) test1: (__unknown_anytype) x;
@end
void test1(B *b) {
  (void) [b test1: (float) 2.0];
}
// CHECK-LABEL: define{{.*}} void @_Z5test1P1B(
// CHECK: call void @objc_msgSend(

