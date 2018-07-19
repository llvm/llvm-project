// RUN: %clang_cc1 -fms-extensions -triple i686-pc-windows-msvc %s -emit-llvm -o - | FileCheck %s

struct A {
  virtual void __fastcall f(int a, int b);
};
void (__fastcall A::*doit())(int, int) {
  return &A::f;
}

// CHECK: define linkonce_odr x86_fastcallcc void @"\01??_9A@@$BA@AI"(%struct.A* inreg %this, ...) {{.*}} comdat align 2 {
// CHECK: [[VPTR:%.*]] = getelementptr inbounds void (%struct.A*, ...)*, void (%struct.A*, ...)** %{{.*}}, i64 0
// CHECK: [[CALLEE:%.*]] = load void (%struct.A*, ...)*, void (%struct.A*, ...)** [[VPTR]]
// CHECK: musttail call x86_fastcallcc void (%struct.A*, ...) [[CALLEE]](%struct.A* inreg %{{.*}}, ...)
// CHECK: ret void
// CHECK: }
