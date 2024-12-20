// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHCEK: %class.CB = type { float, double }
// CHECK: @CB.cb = external constant target("dx.CBuffer", %class.CB, 176, 16, 168)
cbuffer CB : register(b1, space3) {
  float a : packoffset(c1.x);
  double b : packoffset(c10.z);
}

float foo() {
// CHECK: %[[HANDLE1:[0-9]+]] = load target("dx.CBuffer", %class.CB, 176, 16, 168), ptr @CB.cb, align 4
// CHECK: %[[PTR1:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.CBs_176_16_168t(target("dx.CBuffer", %class.CB, 176, 16, 168) %[[HANDLE1]], i32 0)
// CHECK: %a = getelementptr %class.CB, ptr %[[PTR1]], i32 0, i32 0
// CHECK: load float, ptr %a, align 4

// CHECK: %[[HANDLE2:[0-9]+]] = load target("dx.CBuffer", %class.CB, 176, 16, 168), ptr @CB.cb, align 4
// CHECK: %[[PTR2:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.CBs_176_16_168t(target("dx.CBuffer", %class.CB, 176, 16, 168) %[[HANDLE2]], i32 0)
// CHECK: %b = getelementptr %class.CB, ptr %[[PTR2]], i32 0, i32 1
// CHECK: load double, ptr %b, align 8
  return a + b;
}

// CHECK: define internal void @_init_resource_bindings() {
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB_h = call target("dx.CBuffer", %class.CB, 176, 16, 168)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s_class.CBs_176_16_168t(i32 3, i32 1, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", %class.CB, 176, 16, 168) %CB_h, ptr @CB.cb, align 4

[numthreads(4,1,1)]
void main() {
  foo();
}
