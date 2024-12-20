// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: %class.CB = type { float, double }
// CHECK: %class.ParticleLife = type { i32, i32, <2 x float>, float, float }

// CHECK: @CB.cb = external constant target("dx.CBuffer", %class.CB, 16, 0, 8)
cbuffer CB : register(b0, space2) {
  float a;
  double b;
}

// CHECK: @ParticleLife.cb = external constant target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20)
cbuffer ParticleLife : register(b2, space1) {
  uint ParticleTypeMax;
  uint NumParticles;
  float2 WorldSize;
  float Friction;
  float ForceMultipler;
}

float foo() {
// CHECK: %[[HANDLE1:[0-9]+]] = load target("dx.CBuffer", %class.CB, 16, 0, 8), ptr @CB.cb, align 4
// CHECK: %[[PTR1:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.CBs_16_0_8t(target("dx.CBuffer", %class.CB, 16, 0, 8) %[[HANDLE1]], i32 0)
// CHECK: %a = getelementptr %class.CB, ptr %[[PTR1]], i32 0, i32 0
// CHECK: load float, ptr %a, align 4

// CHECK: %[[HANDLE2:[0-9]+]] = load target("dx.CBuffer", %class.CB, 16, 0, 8), ptr @CB.cb, align 4  
// CHECK: %[[PTR2:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.CBs_16_0_8t(target("dx.CBuffer", %class.CB, 16, 0, 8) %[[HANDLE2]], i32 0)
// CHECK: %b = getelementptr %class.CB, ptr %[[PTR2]], i32 0, i32 1
// CHECK: load double, ptr %b, align 8
  
// CHECK: %[[HANDLE3:[0-9]+]] = load target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20), ptr @ParticleLife.cb, align 4
// CHECK: %[[PTR3:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.ParticleLifes_24_0_4_8_16_20t(target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20) %[[HANDLE3]], i32 0)
// CHECK: %ParticleTypeMax = getelementptr %class.ParticleLife, ptr %[[PTR3]], i32 0, i32 0
// CHECK: load i32, ptr %ParticleTypeMax, align 4

// CHECK: %[[HANDLE4:[0-9]+]] = load target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20), ptr @ParticleLife.cb, align 4
// CHECK: %[[PTR4:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.ParticleLifes_24_0_4_8_16_20t(target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20) %[[HANDLE4]], i32 0)
// CHECK: %WorldSize = getelementptr %class.ParticleLife, ptr %[[PTR4]], i32 0, i32 2
// CHECK: %[[VEC:[0-9]+]] = load <2 x float>, ptr %WorldSize, align 8
// CHECK: extractelement <2 x float> %[[VEC]], i32 1
  
// CHECK: %[[HANDLE5:[0-9]+]] = load target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20), ptr @ParticleLife.cb, align 4
// CHECK: %[[PTR5:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.ParticleLifes_24_0_4_8_16_20t(target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20) %[[HANDLE5]], i32 0)
// CHECK: %ForceMultipler = getelementptr %class.ParticleLife, ptr %[[PTR5]], i32 0, i32 4
// CHECK: %15 = load float, ptr %ForceMultipler, align 4
  return a + b + ParticleTypeMax + WorldSize.y * ForceMultipler;
}

// CHECK: define void @main()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_GLOBAL__sub_I_cbuffer.hlsl()
// CHECK-NEXT: call void @_Z4mainv()
// CHECK-NEXT: ret void

// CHECK: define internal void @_init_resource_bindings() {
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB_h = call target("dx.CBuffer", %class.CB, 16, 0, 8)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s_class.CBs_16_0_8t(i32 2, i32 0, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", %class.CB, 16, 0, 8) %CB_h, ptr @CB.cb, align 4
// CHECK-NEXT: %ParticleLife_h = call target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s_class.ParticleLifes_24_0_4_8_16_20t(i32 1, i32 2, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", %class.ParticleLife, 24, 0, 4, 8, 16, 20) %ParticleLife_h, ptr @ParticleLife.cb, align 4

// CHECK: define internal void @_GLOBAL__sub_I_cbuffer.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_init_resource_bindings()
// CHECK-NEXT: ret void

[numthreads(4,1,1)]
void main() {}
