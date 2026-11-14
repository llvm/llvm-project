// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s --check-prefix=CIR --enable-var-scope
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM --enable-var-scope
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM --enable-var-scope

// Exercise all remaining WMMA load builtins through the common load handler.
// The m16n16k16 FP16/FP32 cases are covered in builtins-nvptx-wmma.cu.
// LLVM checks are shared with classic codegen and cover scalar returns,
// aggregate register counts/types, and representative fragment stores.

#define DEVICE extern "C" __attribute__((device))

// CIR-LABEL: @test_hmma_m32n8k16_ld_a(
// LLVM-LABEL: @test_hmma_m32n8k16_ld_a(
DEVICE void test_hmma_m32n8k16_ld_a(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.a.row.stride.f16"
  // LLVM: %[[R0:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[R0]], 7
  // LLVM: %[[BITS:.*]] = bitcast <2 x half> %[[LAST]] to i32
  // LLVM: store i32 %[[BITS]], ptr %{{.*}}, align 4
  __hmma_m32n8k16_ld_a(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.a.col.stride.f16"
  // LLVM: %[[R1:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  __hmma_m32n8k16_ld_a(dst, src, stride, 1);

}

// CIR-LABEL: @test_hmma_m32n8k16_ld_b(
// LLVM-LABEL: @test_hmma_m32n8k16_ld_b(
DEVICE void test_hmma_m32n8k16_ld_b(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.b.row.stride.f16"
  // LLVM: %[[R0:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[R0]], 7
  // LLVM: %[[BITS:.*]] = bitcast <2 x half> %[[LAST]] to i32
  // LLVM: store i32 %[[BITS]], ptr %{{.*}}, align 4
  __hmma_m32n8k16_ld_b(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.b.col.stride.f16"
  // LLVM: %[[R1:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  __hmma_m32n8k16_ld_b(dst, src, stride, 1);

}

// CIR-LABEL: @test_hmma_m32n8k16_ld_c_f16(
// LLVM-LABEL: @test_hmma_m32n8k16_ld_c_f16(
DEVICE void test_hmma_m32n8k16_ld_c_f16(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.c.row.stride.f16"
  // LLVM: %[[R0:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[LAST:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[R0]], 3
  // LLVM: %[[BITS:.*]] = bitcast <2 x half> %[[LAST]] to i32
  // LLVM: store i32 %[[BITS]], ptr %{{.*}}, align 4
  __hmma_m32n8k16_ld_c_f16(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.c.col.stride.f16"
  // LLVM: %[[R1:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  __hmma_m32n8k16_ld_c_f16(dst, src, stride, 1);

}

// CIR-LABEL: @test_hmma_m32n8k16_ld_c_f32(
// LLVM-LABEL: @test_hmma_m32n8k16_ld_c_f32(
DEVICE void test_hmma_m32n8k16_ld_c_f32(float *dst, const float *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.c.row.stride.f32"
  // LLVM: %[[R0:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.f32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[R0]], 7
  // LLVM: store float %[[LAST]], ptr %{{.*}}, align 4
  __hmma_m32n8k16_ld_c_f32(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.c.col.stride.f32"
  // LLVM: %[[R1:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.f32.p0(ptr %{{.*}}, i32 %{{.*}})
  __hmma_m32n8k16_ld_c_f32(dst, src, stride, 1);

}

// CIR-LABEL: @test_hmma_m8n32k16_ld_a(
// LLVM-LABEL: @test_hmma_m8n32k16_ld_a(
DEVICE void test_hmma_m8n32k16_ld_a(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.a.row.stride.f16"
  // LLVM: %[[R0:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[R0]], 7
  // LLVM: %[[BITS:.*]] = bitcast <2 x half> %[[LAST]] to i32
  // LLVM: store i32 %[[BITS]], ptr %{{.*}}, align 4
  __hmma_m8n32k16_ld_a(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.a.col.stride.f16"
  // LLVM: %[[R1:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  __hmma_m8n32k16_ld_a(dst, src, stride, 1);

}

// CIR-LABEL: @test_hmma_m8n32k16_ld_b(
// LLVM-LABEL: @test_hmma_m8n32k16_ld_b(
DEVICE void test_hmma_m8n32k16_ld_b(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.b.row.stride.f16"
  // LLVM: %[[R0:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[R0]], 7
  // LLVM: %[[BITS:.*]] = bitcast <2 x half> %[[LAST]] to i32
  // LLVM: store i32 %[[BITS]], ptr %{{.*}}, align 4
  __hmma_m8n32k16_ld_b(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.b.col.stride.f16"
  // LLVM: %[[R1:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  __hmma_m8n32k16_ld_b(dst, src, stride, 1);

}

// CIR-LABEL: @test_hmma_m8n32k16_ld_c_f16(
// LLVM-LABEL: @test_hmma_m8n32k16_ld_c_f16(
DEVICE void test_hmma_m8n32k16_ld_c_f16(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.c.row.stride.f16"
  // LLVM: %[[R0:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[LAST:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[R0]], 3
  // LLVM: %[[BITS:.*]] = bitcast <2 x half> %[[LAST]] to i32
  // LLVM: store i32 %[[BITS]], ptr %{{.*}}, align 4
  __hmma_m8n32k16_ld_c_f16(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.c.col.stride.f16"
  // LLVM: %[[R1:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
  __hmma_m8n32k16_ld_c_f16(dst, src, stride, 1);

}

// CIR-LABEL: @test_hmma_m8n32k16_ld_c_f32(
// LLVM-LABEL: @test_hmma_m8n32k16_ld_c_f32(
DEVICE void test_hmma_m8n32k16_ld_c_f32(float *dst, const float *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.c.row.stride.f32"
  // LLVM: %[[R0:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.f32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[R0]], 7
  // LLVM: store float %[[LAST]], ptr %{{.*}}, align 4
  __hmma_m8n32k16_ld_c_f32(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.c.col.stride.f32"
  // LLVM: %[[R1:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.f32.p0(ptr %{{.*}}, i32 %{{.*}})
  __hmma_m8n32k16_ld_c_f32(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m16n16k16_ld_a_s8(
// LLVM-LABEL: @test_imma_m16n16k16_ld_a_s8(
DEVICE void test_imma_m16n16k16_ld_a_s8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.row.stride.s8"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][0]
  // LLVM: %[[E0_0:.*]] = extractvalue { i32, i32 } %[[R0]], 0
  // LLVM: store i32 %[[E0_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[E0_1:.*]] = extractvalue { i32, i32 } %[[R0]], 1
  // LLVM: store i32 %[[E0_1]], ptr %{{.*}}, align 4
  __imma_m16n16k16_ld_a_s8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.s8"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R1]][0]
  // LLVM: %[[E1_0:.*]] = extractvalue { i32, i32 } %[[R1]], 0
  // LLVM: store i32 %[[E1_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R1]][1]
  // LLVM: %[[E1_1:.*]] = extractvalue { i32, i32 } %[[R1]], 1
  // LLVM: store i32 %[[E1_1]], ptr %{{.*}}, align 4
  __imma_m16n16k16_ld_a_s8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m16n16k16_ld_b_s8(
// LLVM-LABEL: @test_imma_m16n16k16_ld_b_s8(
DEVICE void test_imma_m16n16k16_ld_b_s8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.b.row.stride.s8"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32 } %[[R0]], 1
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m16n16k16_ld_b_s8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.b.col.stride.s8"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m16n16k16_ld_b_s8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m16n16k16_ld_a_u8(
// LLVM-LABEL: @test_imma_m16n16k16_ld_a_u8(
DEVICE void test_imma_m16n16k16_ld_a_u8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.row.stride.u8"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32 } %[[R0]], 1
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m16n16k16_ld_a_u8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.u8"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m16n16k16_ld_a_u8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m16n16k16_ld_b_u8(
// LLVM-LABEL: @test_imma_m16n16k16_ld_b_u8(
DEVICE void test_imma_m16n16k16_ld_b_u8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.b.row.stride.u8"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32 } %[[R0]], 1
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m16n16k16_ld_b_u8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.b.col.stride.u8"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m16n16k16_ld_b_u8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m16n16k16_ld_c(
// LLVM-LABEL: @test_imma_m16n16k16_ld_c(
DEVICE void test_imma_m16n16k16_ld_c(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.row.stride.s32"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][0]
  // LLVM: %[[E0_0:.*]] = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %[[R0]], 0
  // LLVM: store i32 %[[E0_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[E0_7:.*]] = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %[[R0]], 7
  // LLVM: store i32 %[[E0_7]], ptr %{{.*}}, align 4
  __imma_m16n16k16_ld_c(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.col.stride.s32"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R1]][0]
  // LLVM: %[[E1_0:.*]] = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %[[R1]], 0
  // LLVM: store i32 %[[E1_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R1]][7]
  // LLVM: %[[E1_7:.*]] = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %[[R1]], 7
  // LLVM: store i32 %[[E1_7]], ptr %{{.*}}, align 4
  __imma_m16n16k16_ld_c(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m32n8k16_ld_a_s8(
// LLVM-LABEL: @test_imma_m32n8k16_ld_a_s8(
DEVICE void test_imma_m32n8k16_ld_a_s8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.a.row.stride.s8"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][0]
  // LLVM: %[[E0_0:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 0
  // LLVM: store i32 %[[E0_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[E0_3:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 3
  // LLVM: store i32 %[[E0_3]], ptr %{{.*}}, align 4
  __imma_m32n8k16_ld_a_s8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.a.col.stride.s8"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R1]][0]
  // LLVM: %[[E1_0:.*]] = extractvalue { i32, i32, i32, i32 } %[[R1]], 0
  // LLVM: store i32 %[[E1_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R1]][3]
  // LLVM: %[[E1_3:.*]] = extractvalue { i32, i32, i32, i32 } %[[R1]], 3
  // LLVM: store i32 %[[E1_3]], ptr %{{.*}}, align 4
  __imma_m32n8k16_ld_a_s8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m32n8k16_ld_b_s8(
// LLVM-LABEL: @test_imma_m32n8k16_ld_b_s8(
DEVICE void test_imma_m32n8k16_ld_b_s8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.b.row.stride.s8"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __imma_m32n8k16_ld_b_s8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.b.col.stride.s8"
  // LLVM: %[[R1:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R1]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R1]], ptr %{{.*}}, align 4
  __imma_m32n8k16_ld_b_s8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m32n8k16_ld_a_u8(
// LLVM-LABEL: @test_imma_m32n8k16_ld_a_u8(
DEVICE void test_imma_m32n8k16_ld_a_u8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.a.row.stride.u8"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 3
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m32n8k16_ld_a_u8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.a.col.stride.u8"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m32n8k16_ld_a_u8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m32n8k16_ld_b_u8(
// LLVM-LABEL: @test_imma_m32n8k16_ld_b_u8(
DEVICE void test_imma_m32n8k16_ld_b_u8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.b.row.stride.u8"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __imma_m32n8k16_ld_b_u8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.b.col.stride.u8"
  // LLVM: %[[R1:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R1]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R1]], ptr %{{.*}}, align 4
  __imma_m32n8k16_ld_b_u8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m32n8k16_ld_c(
// LLVM-LABEL: @test_imma_m32n8k16_ld_c(
DEVICE void test_imma_m32n8k16_ld_c(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.c.row.stride.s32"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %[[R0]], 7
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m32n8k16_ld_c(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.c.col.stride.s32"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m32n8k16_ld_c(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m8n32k16_ld_a_s8(
// LLVM-LABEL: @test_imma_m8n32k16_ld_a_s8(
DEVICE void test_imma_m8n32k16_ld_a_s8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.a.row.stride.s8"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __imma_m8n32k16_ld_a_s8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.a.col.stride.s8"
  // LLVM: %[[R1:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R1]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R1]], ptr %{{.*}}, align 4
  __imma_m8n32k16_ld_a_s8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m8n32k16_ld_b_s8(
// LLVM-LABEL: @test_imma_m8n32k16_ld_b_s8(
DEVICE void test_imma_m8n32k16_ld_b_s8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.b.row.stride.s8"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 3
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m8n32k16_ld_b_s8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.b.col.stride.s8"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.s8.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m8n32k16_ld_b_s8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m8n32k16_ld_a_u8(
// LLVM-LABEL: @test_imma_m8n32k16_ld_a_u8(
DEVICE void test_imma_m8n32k16_ld_a_u8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.a.row.stride.u8"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __imma_m8n32k16_ld_a_u8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.a.col.stride.u8"
  // LLVM: %[[R1:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R1]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R1]], ptr %{{.*}}, align 4
  __imma_m8n32k16_ld_a_u8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m8n32k16_ld_b_u8(
// LLVM-LABEL: @test_imma_m8n32k16_ld_b_u8(
DEVICE void test_imma_m8n32k16_ld_b_u8(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.b.row.stride.u8"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 3
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m8n32k16_ld_b_u8(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.b.col.stride.u8"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.u8.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m8n32k16_ld_b_u8(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m8n32k16_ld_c(
// LLVM-LABEL: @test_imma_m8n32k16_ld_c(
DEVICE void test_imma_m8n32k16_ld_c(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.c.row.stride.s32"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %[[R0]], 7
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m8n32k16_ld_c(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.c.col.stride.s32"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m8n32k16_ld_c(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m8n8k32_ld_a_s4(
// LLVM-LABEL: @test_imma_m8n8k32_ld_a_s4(
DEVICE void test_imma_m8n8k32_ld_a_s4(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k32.load.a.row.stride.s4"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n8k32.load.a.row.stride.s4.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __imma_m8n8k32_ld_a_s4(dst, src, stride, 0);

}

// CIR-LABEL: @test_imma_m8n8k32_ld_b_s4(
// LLVM-LABEL: @test_imma_m8n8k32_ld_b_s4(
DEVICE void test_imma_m8n8k32_ld_b_s4(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k32.load.b.col.stride.s4"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n8k32.load.b.col.stride.s4.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __imma_m8n8k32_ld_b_s4(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m8n8k32_ld_a_u4(
// LLVM-LABEL: @test_imma_m8n8k32_ld_a_u4(
DEVICE void test_imma_m8n8k32_ld_a_u4(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k32.load.a.row.stride.u4"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n8k32.load.a.row.stride.u4.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __imma_m8n8k32_ld_a_u4(dst, src, stride, 0);

}

// CIR-LABEL: @test_imma_m8n8k32_ld_b_u4(
// LLVM-LABEL: @test_imma_m8n8k32_ld_b_u4(
DEVICE void test_imma_m8n8k32_ld_b_u4(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k32.load.b.col.stride.u4"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n8k32.load.b.col.stride.u4.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __imma_m8n8k32_ld_b_u4(dst, src, stride, 1);

}

// CIR-LABEL: @test_imma_m8n8k32_ld_c(
// LLVM-LABEL: @test_imma_m8n8k32_ld_c(
DEVICE void test_imma_m8n8k32_ld_c(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k32.load.c.row.stride.s32"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m8n8k32.load.c.row.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32 } %[[R0]], 1
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __imma_m8n8k32_ld_c(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k32.load.c.col.stride.s32"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m8n8k32.load.c.col.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  __imma_m8n8k32_ld_c(dst, src, stride, 1);

}

// CIR-LABEL: @test_bmma_m8n8k128_ld_a_b1(
// LLVM-LABEL: @test_bmma_m8n8k128_ld_a_b1(
DEVICE void test_bmma_m8n8k128_ld_a_b1(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k128.load.a.row.stride.b1"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n8k128.load.a.row.stride.b1.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __bmma_m8n8k128_ld_a_b1(dst, src, stride, 0);

}

// CIR-LABEL: @test_bmma_m8n8k128_ld_b_b1(
// LLVM-LABEL: @test_bmma_m8n8k128_ld_b_b1(
DEVICE void test_bmma_m8n8k128_ld_b_b1(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k128.load.b.col.stride.b1"
  // LLVM: %[[R0:.*]] = call{{.*}} i32 @llvm.nvvm.wmma.m8n8k128.load.b.col.stride.b1.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store i32 %[[R0]], ptr %{{.*}}, align 4
  __bmma_m8n8k128_ld_b_b1(dst, src, stride, 1);

}

// CIR-LABEL: @test_bmma_m8n8k128_ld_c(
// LLVM-LABEL: @test_bmma_m8n8k128_ld_c(
DEVICE void test_bmma_m8n8k128_ld_c(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k128.load.c.row.stride.s32"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m8n8k128.load.c.row.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32 } %[[R0]], 1
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __bmma_m8n8k128_ld_c(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k128.load.c.col.stride.s32"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m8n8k128.load.c.col.stride.s32.p0(ptr %{{.*}}, i32 %{{.*}})
  __bmma_m8n8k128_ld_c(dst, src, stride, 1);

}

// CIR-LABEL: @test_dmma_m8n8k4_ld_a(
// LLVM-LABEL: @test_dmma_m8n8k4_ld_a(
DEVICE void test_dmma_m8n8k4_ld_a(double *dst, const double *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k4.load.a.row.stride.f64"
  // LLVM: %[[R0:.*]] = call{{.*}} double @llvm.nvvm.wmma.m8n8k4.load.a.row.stride.f64.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store double %[[R0]], ptr %{{.*}}, align 4
  __dmma_m8n8k4_ld_a(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k4.load.a.col.stride.f64"
  // LLVM: %[[R1:.*]] = call{{.*}} double @llvm.nvvm.wmma.m8n8k4.load.a.col.stride.f64.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R1]],
  // LLVM-NOT: extractvalue
  // LLVM: store double %[[R1]], ptr %{{.*}}, align 4
  __dmma_m8n8k4_ld_a(dst, src, stride, 1);

}

// CIR-LABEL: @test_dmma_m8n8k4_ld_b(
// LLVM-LABEL: @test_dmma_m8n8k4_ld_b(
DEVICE void test_dmma_m8n8k4_ld_b(double *dst, const double *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k4.load.b.row.stride.f64"
  // LLVM: %[[R0:.*]] = call{{.*}} double @llvm.nvvm.wmma.m8n8k4.load.b.row.stride.f64.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R0]],
  // LLVM-NOT: extractvalue
  // LLVM: store double %[[R0]], ptr %{{.*}}, align 4
  __dmma_m8n8k4_ld_b(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k4.load.b.col.stride.f64"
  // LLVM: %[[R1:.*]] = call{{.*}} double @llvm.nvvm.wmma.m8n8k4.load.b.col.stride.f64.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR-NEXT: cir.store align(4) %[[R1]],
  // LLVM-NOT: extractvalue
  // LLVM: store double %[[R1]], ptr %{{.*}}, align 4
  __dmma_m8n8k4_ld_b(dst, src, stride, 1);

}

// CIR-LABEL: @test_dmma_m8n8k4_ld_c(
// LLVM-LABEL: @test_dmma_m8n8k4_ld_c(
DEVICE void test_dmma_m8n8k4_ld_c(double *dst, const double *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k4.load.c.row.stride.f64"
  // LLVM: %[[R0:.*]] = call{{.*}} { double, double } @llvm.nvvm.wmma.m8n8k4.load.c.row.stride.f64.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][0]
  // LLVM: %[[E0_0:.*]] = extractvalue { double, double } %[[R0]], 0
  // LLVM: store double %[[E0_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[E0_1:.*]] = extractvalue { double, double } %[[R0]], 1
  // LLVM: store double %[[E0_1]], ptr %{{.*}}, align 4
  __dmma_m8n8k4_ld_c(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n8k4.load.c.col.stride.f64"
  // LLVM: %[[R1:.*]] = call{{.*}} { double, double } @llvm.nvvm.wmma.m8n8k4.load.c.col.stride.f64.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R1]][0]
  // LLVM: %[[E1_0:.*]] = extractvalue { double, double } %[[R1]], 0
  // LLVM: store double %[[E1_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R1]][1]
  // LLVM: %[[E1_1:.*]] = extractvalue { double, double } %[[R1]], 1
  // LLVM: store double %[[E1_1]], ptr %{{.*}}, align 4
  __dmma_m8n8k4_ld_c(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_bf16_m16n16k16_ld_a(
// LLVM-LABEL: @test_mma_bf16_m16n16k16_ld_a(
DEVICE void test_mma_bf16_m16n16k16_ld_a(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.row.stride.bf16"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][0]
  // LLVM: %[[E0_0:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 0
  // LLVM: store i32 %[[E0_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[E0_3:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 3
  // LLVM: store i32 %[[E0_3]], ptr %{{.*}}, align 4
  __mma_bf16_m16n16k16_ld_a(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.bf16"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R1]][0]
  // LLVM: %[[E1_0:.*]] = extractvalue { i32, i32, i32, i32 } %[[R1]], 0
  // LLVM: store i32 %[[E1_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R1]][3]
  // LLVM: %[[E1_3:.*]] = extractvalue { i32, i32, i32, i32 } %[[R1]], 3
  // LLVM: store i32 %[[E1_3]], ptr %{{.*}}, align 4
  __mma_bf16_m16n16k16_ld_a(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_bf16_m16n16k16_ld_b(
// LLVM-LABEL: @test_mma_bf16_m16n16k16_ld_b(
DEVICE void test_mma_bf16_m16n16k16_ld_b(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.b.row.stride.bf16"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 3
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __mma_bf16_m16n16k16_ld_b(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.b.col.stride.bf16"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  __mma_bf16_m16n16k16_ld_b(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_bf16_m32n8k16_ld_a(
// LLVM-LABEL: @test_mma_bf16_m32n8k16_ld_a(
DEVICE void test_mma_bf16_m32n8k16_ld_a(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.a.row.stride.bf16"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %[[R0]], 7
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __mma_bf16_m32n8k16_ld_a(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.a.col.stride.bf16"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  __mma_bf16_m32n8k16_ld_a(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_bf16_m32n8k16_ld_b(
// LLVM-LABEL: @test_mma_bf16_m32n8k16_ld_b(
DEVICE void test_mma_bf16_m32n8k16_ld_b(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.b.row.stride.bf16"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32 } %[[R0]], 1
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __mma_bf16_m32n8k16_ld_b(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m32n8k16.load.b.col.stride.bf16"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  __mma_bf16_m32n8k16_ld_b(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_bf16_m8n32k16_ld_a(
// LLVM-LABEL: @test_mma_bf16_m8n32k16_ld_a(
DEVICE void test_mma_bf16_m8n32k16_ld_a(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.a.row.stride.bf16"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][1]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32 } %[[R0]], 1
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __mma_bf16_m8n32k16_ld_a(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.a.col.stride.bf16"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  __mma_bf16_m8n32k16_ld_a(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_bf16_m8n32k16_ld_b(
// LLVM-LABEL: @test_mma_bf16_m8n32k16_ld_b(
DEVICE void test_mma_bf16_m8n32k16_ld_b(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.b.row.stride.bf16"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %[[R0]], 7
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __mma_bf16_m8n32k16_ld_b(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m8n32k16.load.b.col.stride.bf16"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.bf16.p0(ptr %{{.*}}, i32 %{{.*}})
  __mma_bf16_m8n32k16_ld_b(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_tf32_m16n16k8_ld_a(
// LLVM-LABEL: @test_mma_tf32_m16n16k8_ld_a(
DEVICE void test_mma_tf32_m16n16k8_ld_a(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.load.a.row.stride.tf32"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.row.stride.tf32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][0]
  // LLVM: %[[E0_0:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 0
  // LLVM: store i32 %[[E0_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[E0_3:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 3
  // LLVM: store i32 %[[E0_3]], ptr %{{.*}}, align 4
  __mma_tf32_m16n16k8_ld_a(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.load.a.col.stride.tf32"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.col.stride.tf32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R1]][0]
  // LLVM: %[[E1_0:.*]] = extractvalue { i32, i32, i32, i32 } %[[R1]], 0
  // LLVM: store i32 %[[E1_0]], ptr %{{.*}}, align 4
  // CIR: cir.extract_member %[[R1]][3]
  // LLVM: %[[E1_3:.*]] = extractvalue { i32, i32, i32, i32 } %[[R1]], 3
  // LLVM: store i32 %[[E1_3]], ptr %{{.*}}, align 4
  __mma_tf32_m16n16k8_ld_a(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_tf32_m16n16k8_ld_b(
// LLVM-LABEL: @test_mma_tf32_m16n16k8_ld_b(
DEVICE void test_mma_tf32_m16n16k8_ld_b(int *dst, const int *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.load.b.row.stride.tf32"
  // LLVM: %[[R0:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.b.row.stride.tf32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][3]
  // LLVM: %[[LAST:.*]] = extractvalue { i32, i32, i32, i32 } %[[R0]], 3
  // LLVM: store i32 %[[LAST]], ptr %{{.*}}, align 4
  __mma_tf32_m16n16k8_ld_b(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.load.b.col.stride.tf32"
  // LLVM: %[[R1:.*]] = call{{.*}} { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.b.col.stride.tf32.p0(ptr %{{.*}}, i32 %{{.*}})
  __mma_tf32_m16n16k8_ld_b(dst, src, stride, 1);

}

// CIR-LABEL: @test_mma_tf32_m16n16k8_ld_c(
// LLVM-LABEL: @test_mma_tf32_m16n16k8_ld_c(
DEVICE void test_mma_tf32_m16n16k8_ld_c(float *dst, const float *src, unsigned stride) {
  // CIR: %[[R0:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.load.c.row.stride.f32"
  // LLVM: %[[R0:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.load.c.row.stride.f32.p0(ptr %{{.*}}, i32 %{{.*}})
  // CIR: cir.extract_member %[[R0]][7]
  // LLVM: %[[LAST:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[R0]], 7
  // LLVM: store float %[[LAST]], ptr %{{.*}}, align 4
  __mma_tf32_m16n16k8_ld_c(dst, src, stride, 0);

  // CIR: %[[R1:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k8.load.c.col.stride.f32"
  // LLVM: %[[R1:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.load.c.col.stride.f32.p0(ptr %{{.*}}, i32 %{{.*}})
  __mma_tf32_m16n16k8_ld_c(dst, src, stride, 1);

}
