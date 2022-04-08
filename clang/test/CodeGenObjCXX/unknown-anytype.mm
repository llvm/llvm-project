// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -fdebugger-support -funknown-anytype -emit-llvm -o - %s | FileCheck %s

// rdar://13025708

@interface A @end
void test0(A *a) {
  (void) [a test0: (float) 2.0];
}
// CHECK-LABEL: define{{.*}} void @_Z5test0P1A(
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, float)*)(

@interface B
- (void) test1: (__unknown_anytype) x;
@end
void test1(B *b) {
  (void) [b test1: (float) 2.0];
}
// CHECK-LABEL: define{{.*}} void @_Z5test1P1B(
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, float)*)(

