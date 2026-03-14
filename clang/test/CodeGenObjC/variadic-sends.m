// RUN: %clang_cc1 -triple i386-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-X86-32 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-X86-64 %s

@interface A
-(void) im0;
-(void) im1: (int) x;
-(void) im2: (int) x, ...;
@end

void f0(A *a) {
  // CHECK-X86-32: call void @objc_msgSend
  // CHECK-X86-64: call void @objc_msgSend
  [a im0];
}

void f1(A *a) {
  // CHECK-X86-32: call void @objc_msgSend
  // CHECK-X86-64: call void @objc_msgSend
  [a im1: 1];
}

void f2(A *a) {
  // CHECK-X86-32: call void (ptr, ptr, i32, ...) @objc_msgSend
  // CHECK-X86-64: call void (ptr, ptr, i32, ...) @objc_msgSend
  [a im2: 1, 2];
}

@interface B : A @end
@implementation B : A
-(void) foo {
  // CHECK-X86-32: call void @objc_msgSendSuper
  // CHECK-X86-64: call void @objc_msgSendSuper
  [super im1: 1];
}
-(void) bar {
  // CHECK-X86-32: call void (ptr, ptr, i32, ...) @objc_msgSendSuper
  // CHECK-X86-64: call void (ptr, ptr, i32, ...) @objc_msgSendSuper
  [super im2: 1, 2];
}

@end
