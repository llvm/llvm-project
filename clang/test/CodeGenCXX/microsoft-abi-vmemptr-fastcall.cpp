// RUN: %clang_cc1 -fms-extensions -triple i686-pc-windows-msvc %s -emit-llvm -o - | FileCheck %s

struct A {
  virtual void __fastcall f(int a, int b);
};
void (__fastcall A::*doit())(int, int) {
  return &A::f;
}

// CHECK: define linkonce_odr x86_fastcallcc void @"??_9A@@$BA@AI"(ptr inreg noundef %this, ...) {{.*}} comdat align 2 {
// CHECK: [[VPTR:%.*]] = getelementptr inbounds ptr, ptr %{{.*}}, i64 0
// CHECK: [[CALLEE:%.*]] = load ptr, ptr [[VPTR]]
// CHECK: musttail call x86_fastcallcc void (ptr, ...) [[CALLEE]](ptr inreg noundef %{{.*}}, ...)
// CHECK: ret void
// CHECK: }
