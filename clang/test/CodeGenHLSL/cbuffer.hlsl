// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: @CB1.cb = external constant target("dx.CBuffer", { float, double }, 16, 0, 8)
cbuffer CB1 : register(b0, space2) {
  float a;
  double b;
}

// CHECK: @ParticleLifeCB.cb = external constant target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20)
cbuffer ParticleLifeCB : register(b2, space1) {
  uint ParticleTypeMax;
  uint NumParticles;
  float2 WorldSize;
  float Friction;
  float ForceMultipler;
}

float foo() {
// CHECK: %[[HANDLE1:[0-9]+]] = load target("dx.CBuffer", { float, double }, 16, 0, 8), ptr @CB1.cb, align 4
// CHECK: %[[PTR1:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_sl_f32f64s_16_0_8t(target("dx.CBuffer", { float, double }, 16, 0, 8) %[[HANDLE1]], i32 0)
// CHECK: load float, ptr %[[PTR1]], align 4

// CHECK: %[[HANDLE2:[0-9]+]] = load target("dx.CBuffer", { float, double }, 16, 0, 8), ptr @CB1.cb, align 4  
// CHECK: %[[PTR2:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_sl_f32f64s_16_0_8t(target("dx.CBuffer", { float, double }, 16, 0, 8) %[[HANDLE2]], i32 4)
// CHECK: load double, ptr %[[PTR2]], align 8
  
// CHECK: %[[HANDLE3:[0-9]+]] = load target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20), ptr @ParticleLifeCB.cb, align 4
// CHECK: %[[PTR3:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_sl_i32i32v2f32f32f32s_24_0_4_8_16_20t(
// CHECK-SAME: target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20) %[[HANDLE3]], i32 0)
// CHECK: load i32, ptr %[[PTR3]], align 4

// CHECK: %[[HANDLE4:[0-9]+]] = load target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20), ptr @ParticleLifeCB.cb, align 4
// CHECK: %[[PTR4:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_sl_i32i32v2f32f32f32s_24_0_4_8_16_20t(
// CHECK-SAME: target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20) %[[HANDLE4]], i32 8)
// CHECK: load <2 x float>, ptr %[[PTR4]], align 8

// CHECK: %[[HANDLE5:[0-9]+]] = load target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20), ptr @ParticleLifeCB.cb, align 4
// CHECK: %[[PTR5:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_sl_i32i32v2f32f32f32s_24_0_4_8_16_20t(
// CHECK-SAME: target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20) %[[HANDLE5]], i32 20)
// CHECK: load float, ptr %[[PTR5]], align 4
  return a + b + ParticleTypeMax + WorldSize.y * ForceMultipler;
}

// CHECK: define void @main()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_GLOBAL__sub_I_cbuffer.hlsl()
// CHECK-NEXT: call void @_Z4mainv()
// CHECK-NEXT: ret void

// CHECK: define internal void @_init_resource_bindings() {
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB1_h = call target("dx.CBuffer", { float, double }, 16, 0, 8)
// CHECK-SAME: @llvm.dx.handle.fromBinding.tdx.CBuffer_sl_f32f64s_16_0_8t(i32 2, i32 0, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", { float, double }, 16, 0, 8) %CB1_h, ptr @CB1.cb, align 4
// CHECK-NEXT: %ParticleLifeCB_h = call target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20)
// CHECK-SAME: @llvm.dx.handle.fromBinding.tdx.CBuffer_sl_i32i32v2f32f32f32s_24_0_4_8_16_20t(i32 1, i32 2, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", { i32, i32, <2 x float>, float, float }, 24, 0, 4, 8, 16, 20) %ParticleLifeCB_h, ptr @ParticleLifeCB.cb, align 4

// CHECK: define internal void @_GLOBAL__sub_I_cbuffer.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_init_resource_bindings()
// CHECK-NEXT: ret void

[numthreads(4,1,1)]
void main() {}
