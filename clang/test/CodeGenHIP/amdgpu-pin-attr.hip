// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx950 -x hip \
// RUN:   -fcuda-is-device -emit-llvm -o - %s | FileCheck %s

// The amdgpu_pin_{agpr,vgpr} attribute wraps every store to the local in the
// corresponding llvm.amdgcn.pin.* intrinsic (chunked to i32-based widths).

typedef float float2 __attribute__((ext_vector_type(2)));

// CHECK-LABEL: define{{.*}}pin_agpr
// CHECK: call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %{{[0-9]+}}, i32 8)
__attribute__((device)) void pin_agpr(float2 *out, float2 in) {
  __attribute__((amdgpu_pin_agpr(8))) float2 x;
  x = in;
  *out = x;
}

// CHECK-LABEL: define{{.*}}pin_vgpr
// CHECK: call <2 x i32> @llvm.amdgcn.pin.vgpr.v2i32(<2 x i32> %{{[0-9]+}}, i32 4)
__attribute__((device)) void pin_vgpr(float2 *out, float2 in) {
  __attribute__((amdgpu_pin_vgpr(4))) float2 x;
  x = in;
  *out = x;
}

// A value wider than the largest pin width is decomposed into i32 chunks of
// 16/8/4/2/1 dwords at consecutive register numbers (12 dwords -> 8 + 4).
typedef float float12 __attribute__((ext_vector_type(12)));
// CHECK-LABEL: define{{.*}}pin_wide
// CHECK: call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %{{[0-9]+}}, i32 0)
// CHECK: call <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32> %{{[0-9]+}}, i32 8)
__attribute__((device)) void pin_wide(float12 *out, float12 in) {
  __attribute__((amdgpu_pin_vgpr(0))) float12 x;
  x = in;
  *out = x;
}

// A constant-expression argument (here via a template parameter) is accepted and
// evaluated at instantiation.
template <int N>
__attribute__((device)) void pin_tmpl(float2 *out, float2 in) {
  __attribute__((amdgpu_pin_agpr(N + 4))) float2 x;
  x = in;
  *out = x;
}
// CHECK-LABEL: define{{.*}}pin_tmpl
// CHECK: call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %{{[0-9]+}}, i32 12)
__attribute__((device)) void use_tmpl(float2 *out, float2 in) {
  pin_tmpl<8>(out, in);
}
