// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s --check-prefix=CIR --enable-var-scope
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM --enable-var-scope
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM --enable-var-scope

// Cover the SM70/PTX60 m16n16k16 floating-point WMMA load builtins. FP16
// fragments use i32 storage slots and <2 x half> intrinsic registers.
// The LLVM checks are shared with classic codegen.

#define DEVICE extern "C" __attribute__((device))

// CIR-LABEL: @load_a_f16_row(
// LLVM-LABEL: @load_a_f16_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[SRC:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[STRIDE:.*]] = load i32, ptr %{{.*}}, align 4
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.row.stride.f16"
// CIR: %[[E0:.*]] = cir.extract_member %[[RESULT]][0]
// CIR: cir.cast bitcast %[[E0]] : !cir.vector<2 x !cir.f16> -> !s32i
// CIR: cir.extract_member %[[RESULT]][7]
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr %[[SRC]], i32 %[[STRIDE]])
// LLVM: %[[E0:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 0
// LLVM: %[[BITS0:.*]] = bitcast <2 x half> %[[E0]] to i32
// LLVM: store i32 %[[BITS0]], ptr %[[P0:.*]], align 4
// LLVM: %[[E1:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 1
// LLVM: %[[BITS1:.*]] = bitcast <2 x half> %[[E1]] to i32
// LLVM: %[[P1:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 1
// LLVM: store i32 %[[BITS1]], ptr %[[P1]], align 4
// LLVM: %[[E2:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 2
// LLVM: %[[BITS2:.*]] = bitcast <2 x half> %[[E2]] to i32
// LLVM: %[[P2:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 2
// LLVM: store i32 %[[BITS2]], ptr %[[P2]], align 4
// LLVM: %[[E3:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 3
// LLVM: %[[BITS3:.*]] = bitcast <2 x half> %[[E3]] to i32
// LLVM: %[[P3:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 3
// LLVM: store i32 %[[BITS3]], ptr %[[P3]], align 4
// LLVM: %[[E4:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 4
// LLVM: %[[BITS4:.*]] = bitcast <2 x half> %[[E4]] to i32
// LLVM: %[[P4:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 4
// LLVM: store i32 %[[BITS4]], ptr %[[P4]], align 4
// LLVM: %[[E5:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 5
// LLVM: %[[BITS5:.*]] = bitcast <2 x half> %[[E5]] to i32
// LLVM: %[[P5:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 5
// LLVM: store i32 %[[BITS5]], ptr %[[P5]], align 4
// LLVM: %[[E6:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 6
// LLVM: %[[BITS6:.*]] = bitcast <2 x half> %[[E6]] to i32
// LLVM: %[[P6:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 6
// LLVM: store i32 %[[BITS6]], ptr %[[P6]], align 4
// LLVM: %[[E7:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 7
// LLVM: %[[BITS7:.*]] = bitcast <2 x half> %[[E7]] to i32
// LLVM: %[[P7:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 7
// LLVM: store i32 %[[BITS7]], ptr %[[P7]], align 4
// LLVM: ret void
DEVICE void load_a_f16_row(int *dst, const int *src, unsigned stride) {
  __hmma_m16n16k16_ld_a(dst, src, stride, 0);
}

// CIR-LABEL: @load_a_f16_col(
// LLVM-LABEL: @load_a_f16_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.a.col.stride.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
DEVICE void load_a_f16_col(int *dst, const int *src, unsigned stride) {
  __hmma_m16n16k16_ld_a(dst, src, stride, 1);
}

// CIR-LABEL: @load_b_f16_row(
// LLVM-LABEL: @load_b_f16_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[SRC:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[STRIDE:.*]] = load i32, ptr %{{.*}}, align 4
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.b.row.stride.f16"
// CIR: %[[E0:.*]] = cir.extract_member %[[RESULT]][0]
// CIR: cir.cast bitcast %[[E0]] : !cir.vector<2 x !cir.f16> -> !s32i
// CIR: cir.extract_member %[[RESULT]][7]
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr %[[SRC]], i32 %[[STRIDE]])
// LLVM: %[[E0:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 0
// LLVM: %[[BITS0:.*]] = bitcast <2 x half> %[[E0]] to i32
// LLVM: store i32 %[[BITS0]], ptr %[[P0:.*]], align 4
// LLVM: %[[E1:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 1
// LLVM: %[[BITS1:.*]] = bitcast <2 x half> %[[E1]] to i32
// LLVM: %[[P1:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 1
// LLVM: store i32 %[[BITS1]], ptr %[[P1]], align 4
// LLVM: %[[E2:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 2
// LLVM: %[[BITS2:.*]] = bitcast <2 x half> %[[E2]] to i32
// LLVM: %[[P2:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 2
// LLVM: store i32 %[[BITS2]], ptr %[[P2]], align 4
// LLVM: %[[E3:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 3
// LLVM: %[[BITS3:.*]] = bitcast <2 x half> %[[E3]] to i32
// LLVM: %[[P3:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 3
// LLVM: store i32 %[[BITS3]], ptr %[[P3]], align 4
// LLVM: %[[E4:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 4
// LLVM: %[[BITS4:.*]] = bitcast <2 x half> %[[E4]] to i32
// LLVM: %[[P4:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 4
// LLVM: store i32 %[[BITS4]], ptr %[[P4]], align 4
// LLVM: %[[E5:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 5
// LLVM: %[[BITS5:.*]] = bitcast <2 x half> %[[E5]] to i32
// LLVM: %[[P5:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 5
// LLVM: store i32 %[[BITS5]], ptr %[[P5]], align 4
// LLVM: %[[E6:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 6
// LLVM: %[[BITS6:.*]] = bitcast <2 x half> %[[E6]] to i32
// LLVM: %[[P6:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 6
// LLVM: store i32 %[[BITS6]], ptr %[[P6]], align 4
// LLVM: %[[E7:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 7
// LLVM: %[[BITS7:.*]] = bitcast <2 x half> %[[E7]] to i32
// LLVM: %[[P7:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 7
// LLVM: store i32 %[[BITS7]], ptr %[[P7]], align 4
// LLVM: ret void
DEVICE void load_b_f16_row(int *dst, const int *src, unsigned stride) {
  __hmma_m16n16k16_ld_b(dst, src, stride, 0);
}

// CIR-LABEL: @load_b_f16_col(
// LLVM-LABEL: @load_b_f16_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.b.col.stride.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
DEVICE void load_b_f16_col(int *dst, const int *src, unsigned stride) {
  __hmma_m16n16k16_ld_b(dst, src, stride, 1);
}

// CIR-LABEL: @load_c_f16_row(
// LLVM-LABEL: @load_c_f16_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[SRC:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[STRIDE:.*]] = load i32, ptr %{{.*}}, align 4
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.row.stride.f16"
// CIR: %[[E0:.*]] = cir.extract_member %[[RESULT]][0]
// CIR: cir.cast bitcast %[[E0]] : !cir.vector<2 x !cir.f16> -> !s32i
// CIR: cir.extract_member %[[RESULT]][3]
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f16.p0(ptr %[[SRC]], i32 %[[STRIDE]])
// LLVM: %[[E0:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 0
// LLVM: %[[BITS0:.*]] = bitcast <2 x half> %[[E0]] to i32
// LLVM: store i32 %[[BITS0]], ptr %[[P0:.*]], align 4
// LLVM: %[[E1:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 1
// LLVM: %[[BITS1:.*]] = bitcast <2 x half> %[[E1]] to i32
// LLVM: %[[P1:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 1
// LLVM: store i32 %[[BITS1]], ptr %[[P1]], align 4
// LLVM: %[[E2:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 2
// LLVM: %[[BITS2:.*]] = bitcast <2 x half> %[[E2]] to i32
// LLVM: %[[P2:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 2
// LLVM: store i32 %[[BITS2]], ptr %[[P2]], align 4
// LLVM: %[[E3:.*]] = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %[[RESULT]], 3
// LLVM: %[[BITS3:.*]] = bitcast <2 x half> %[[E3]] to i32
// LLVM: %[[P3:.*]] = getelementptr{{.*}} i32, ptr %[[DST]], i{{32|64}} 3
// LLVM: store i32 %[[BITS3]], ptr %[[P3]], align 4
// LLVM: ret void
DEVICE void load_c_f16_row(int *dst, const int *src, unsigned stride) {
  __hmma_m16n16k16_ld_c_f16(dst, src, stride, 0);
}

// CIR-LABEL: @load_c_f16_col(
// LLVM-LABEL: @load_c_f16_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.col.stride.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f16.p0(ptr %{{.*}}, i32 %{{.*}})
DEVICE void load_c_f16_col(int *dst, const int *src, unsigned stride) {
  __hmma_m16n16k16_ld_c_f16(dst, src, stride, 1);
}

// CIR-LABEL: @load_c_f32_row(
// LLVM-LABEL: @load_c_f32_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[SRC:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[STRIDE:.*]] = load i32, ptr %{{.*}}, align 4
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.row.stride.f32"
// CIR: %[[E0:.*]] = cir.extract_member %[[RESULT]][0]
// CIR: cir.extract_member %[[RESULT]][7]
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p0(ptr %[[SRC]], i32 %[[STRIDE]])
// LLVM: %[[E0:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[RESULT]], 0
// LLVM: store float %[[E0]], ptr %[[P0:.*]], align 4
// LLVM: %[[E1:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[RESULT]], 1
// LLVM: %[[P1:.*]] = getelementptr{{.*}} float, ptr %[[DST]], i{{32|64}} 1
// LLVM: store float %[[E1]], ptr %[[P1]], align 4
// LLVM: %[[E2:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[RESULT]], 2
// LLVM: %[[P2:.*]] = getelementptr{{.*}} float, ptr %[[DST]], i{{32|64}} 2
// LLVM: store float %[[E2]], ptr %[[P2]], align 4
// LLVM: %[[E3:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[RESULT]], 3
// LLVM: %[[P3:.*]] = getelementptr{{.*}} float, ptr %[[DST]], i{{32|64}} 3
// LLVM: store float %[[E3]], ptr %[[P3]], align 4
// LLVM: %[[E4:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[RESULT]], 4
// LLVM: %[[P4:.*]] = getelementptr{{.*}} float, ptr %[[DST]], i{{32|64}} 4
// LLVM: store float %[[E4]], ptr %[[P4]], align 4
// LLVM: %[[E5:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[RESULT]], 5
// LLVM: %[[P5:.*]] = getelementptr{{.*}} float, ptr %[[DST]], i{{32|64}} 5
// LLVM: store float %[[E5]], ptr %[[P5]], align 4
// LLVM: %[[E6:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[RESULT]], 6
// LLVM: %[[P6:.*]] = getelementptr{{.*}} float, ptr %[[DST]], i{{32|64}} 6
// LLVM: store float %[[E6]], ptr %[[P6]], align 4
// LLVM: %[[E7:.*]] = extractvalue { float, float, float, float, float, float, float, float } %[[RESULT]], 7
// LLVM: %[[P7:.*]] = getelementptr{{.*}} float, ptr %[[DST]], i{{32|64}} 7
// LLVM: store float %[[E7]], ptr %[[P7]], align 4
// LLVM: ret void
DEVICE void load_c_f32_row(float *dst, const float *src, unsigned stride) {
  __hmma_m16n16k16_ld_c_f32(dst, src, stride, 0);
}

// CIR-LABEL: @load_c_f32_col(
// LLVM-LABEL: @load_c_f32_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.load.c.col.stride.f32"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p0(ptr %{{.*}}, i32 %{{.*}})
DEVICE void load_c_f32_col(float *dst, const float *src, unsigned stride) {
  __hmma_m16n16k16_ld_c_f32(dst, src, stride, 1);
}
