// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1100 -target-feature +extended-image-insts \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu gfx1100 -target-feature +extended-image-insts \
// RUN:            -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu gfx1100 -target-feature +extended-image-insts \
// RUN:            -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU extended image builtins in OpenCL
//===----------------------------------------------------------------------===//

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef half half4 __attribute__((ext_vector_type(4)));

// CIR-LABEL: @test_image_gather4_lz_2d_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.gather4.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_gather4_lz_2d_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.gather4.lz.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_gather4_lz_2d_v4f32(
// OGCG: call {{.*}}<4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_gather4_lz_2d_v4f32(global float4* out, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(1, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_1d_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.1d" {{.*}} : (!s32i, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_1d_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.1d.{{.*}}(i32 {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_1d_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.1d.{{.*}}(i32 {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_1d_v4f32(global float4* out, float s, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_1d_v4f32_f32(100, s, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_1d_v4f16
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.1d" {{.*}} : (!s32i, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.f16 x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_1d_v4f16(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.1d.{{.*}}(i32 {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_1d_v4f16(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.1d.{{.*}}(i32 {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_1d_v4f16(global half4* out, float s, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_1d_v4f16_f32(100, s, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_l_1d_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.l.1d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_l_1d_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.l.1d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_l_1d_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.l.1d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_l_1d_v4f32(global float4* out, float s, float lod, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_l_1d_v4f32_f32(100, s, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_l_1d_v4f16
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.l.1d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.f16 x 4>
// LLVM: define{{.*}} void @test_image_sample_l_1d_v4f16(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.l.1d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_l_1d_v4f16(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.l.1d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_l_1d_v4f16(global half4* out, float s, float lod, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_l_1d_v4f16_f32(100, s, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_d_1d_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.d.1d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_d_1d_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.d.1d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_d_1d_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.d.1d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_d_1d_v4f32(global float4* out, float dsdx, float dsdy, float s, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_d_1d_v4f32_f32(100, dsdx, dsdy, s, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_d_1d_v4f16
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.d.1d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.f16 x 4>
// LLVM: define{{.*}} void @test_image_sample_d_1d_v4f16(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.d.1d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_d_1d_v4f16(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.d.1d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_d_1d_v4f16(global half4* out, float dsdx, float dsdy, float s, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_d_1d_v4f16_f32(100, dsdx, dsdy, s, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_2d_f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} void @test_image_sample_lz_2d_f32(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.lz.2d.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_2d_f32(
// OGCG: call {{.*}}float @llvm.amdgcn.image.sample.lz.2d.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_2d_f32(global float* out, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_2d_f32_f32(1, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_2d_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_2d_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_2d_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_2d_v4f32(global float4* out, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_2d_v4f32_f32(100, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_2d_v4f16
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.f16 x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_2d_v4f16(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_2d_v4f16(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_2d_v4f16(global half4* out, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_2d_v4f16_f32(100, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_l_2d_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.l.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_l_2d_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.l.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_l_2d_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.l.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_l_2d_v4f32(global float4* out, float s, float t, float lod, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_l_2d_v4f32_f32(10, s, t, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_l_2d_f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.l.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} void @test_image_sample_l_2d_f32(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.l.2d.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_l_2d_f32(
// OGCG: call {{.*}}float @llvm.amdgcn.image.sample.l.2d.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_l_2d_f32(global float* out, float s, float t, float lod, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_l_2d_f32_f32(1, s, t, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_d_2d_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.d.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_d_2d_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.d.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_d_2d_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.d.2d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_d_2d_v4f32(global float4* out, float dsdx, float dtdx, float dsdy, float dtdy, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_d_2d_v4f32_f32(100, dsdx, dtdx, dsdy, dtdy, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_d_2d_f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.d.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} void @test_image_sample_d_2d_f32(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_d_2d_f32(
// OGCG: call {{.*}}float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_d_2d_f32(global float* out, float dsdx, float dtdx, float dsdy, float dtdy, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_d_2d_f32_f32(1, dsdx, dtdx, dsdy, dtdy, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_3d_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.3d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_3d_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.3d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_3d_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.3d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_3d_v4f32(global float4* out, float s, float t, float r, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_3d_v4f32_f32(100, s, t, r, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_3d_v4f16
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.3d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.f16 x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_3d_v4f16(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.3d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_3d_v4f16(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.3d.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_3d_v4f16(global half4* out, float s, float t, float r, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_3d_v4f16_f32(100, s, t, r, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_cube_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.cube" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_cube_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.cube.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_cube_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.cube.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_cube_v4f32(global float4* out, float s, float t, float face, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_cube_v4f32_f32(1, s, t, face, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_cube_v4f16
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.cube" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.f16 x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_cube_v4f16(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.cube.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_cube_v4f16(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.cube.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_cube_v4f16(global half4* out, float s, float t, float face, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_cube_v4f16_f32(100, s, t, face, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_1darray_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.1darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_1darray_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.1darray.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_1darray_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.1darray.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_1darray_v4f32(global float4* out, float s, float slice, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_1darray_v4f32_f32(1, s, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_1darray_v4f16
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.1darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.f16 x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_1darray_v4f16(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.1darray.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_1darray_v4f16(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.1darray.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_1darray_v4f16(global half4* out, float s, float slice, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_1darray_v4f16_f32(100, s, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_2darray_f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} void @test_image_sample_lz_2darray_f32(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.lz.2darray.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_2darray_f32(
// OGCG: call {{.*}}float @llvm.amdgcn.image.sample.lz.2darray.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_2darray_f32(global float* out, float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_2darray_f32_f32(1, s, t, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_2darray_v4f32
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.float x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_2darray_v4f32(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.2darray.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_2darray_v4f32(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.2darray.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_2darray_v4f32(global float4* out, float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_2darray_v4f32_f32(100, s, t, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @test_image_sample_lz_2darray_v4f16
// CIR: cir.llvm.intrinsic "amdgcn.image.sample.lz.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<!s32i x 8>, !cir.vector<!s32i x 4>, !cir.bool, !s32i, !s32i) -> !cir.vector<!cir.f16 x 4>
// LLVM: define{{.*}} void @test_image_sample_lz_2darray_v4f16(
// LLVM: call {{.*}}@llvm.amdgcn.image.sample.lz.2darray.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
// OGCG: define{{.*}} void @test_image_sample_lz_2darray_v4f16(
// OGCG: call {{.*}}@llvm.amdgcn.image.sample.lz.2darray.{{.*}}(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
void test_image_sample_lz_2darray_v4f16(global half4* out, float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  *out = __builtin_amdgcn_image_sample_lz_2darray_v4f16_f32(100, s, t, slice, tex, samp, 0, 120, 110);
}
