// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s --check-prefix=CIR --enable-var-scope
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM --enable-var-scope
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM --enable-var-scope

// Cover the SM70/PTX60 m16n16k16 floating-point WMMA builtins. FP16
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

// CIR-LABEL: @store_f16_row(
// LLVM-LABEL: @store_f16_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[SRC:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[STRIDE:.*]] = load i32, ptr %{{.*}}, align 4
// CIR: cir.cast bitcast {{.*}} : !s32i -> !cir.vector<2 x !cir.f16>
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.store.d.row.stride.f16"
// LLVM: %[[S0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[SR0:.*]] = bitcast i32 %[[S0]] to <2 x half>
// LLVM: %[[SP1:.*]] = getelementptr{{.*}} i32, ptr %[[SRC]], i{{32|64}} 1
// LLVM: %[[S1:.*]] = load i32, ptr %[[SP1]], align 4
// LLVM: %[[SR1:.*]] = bitcast i32 %[[S1]] to <2 x half>
// LLVM: %[[SP2:.*]] = getelementptr{{.*}} i32, ptr %[[SRC]], i{{32|64}} 2
// LLVM: %[[S2:.*]] = load i32, ptr %[[SP2]], align 4
// LLVM: %[[SR2:.*]] = bitcast i32 %[[S2]] to <2 x half>
// LLVM: %[[SP3:.*]] = getelementptr{{.*}} i32, ptr %[[SRC]], i{{32|64}} 3
// LLVM: %[[S3:.*]] = load i32, ptr %[[SP3]], align 4
// LLVM: %[[SR3:.*]] = bitcast i32 %[[S3]] to <2 x half>
// LLVM: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p0(ptr %[[DST]], <2 x half> %[[SR0]], <2 x half> %[[SR1]], <2 x half> %[[SR2]], <2 x half> %[[SR3]], i32 %[[STRIDE]])
// LLVM: ret void
DEVICE void store_f16_row(int *dst, const int *src, unsigned stride) {
  __hmma_m16n16k16_st_c_f16(dst, src, stride, 0);
}

// CIR-LABEL: @store_f16_col(
// LLVM-LABEL: @store_f16_col(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.store.d.col.stride.f16"
// LLVM: call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f16.p0(ptr %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, i32 %{{.*}})
// LLVM: ret void
DEVICE void store_f16_col(int *dst, const int *src, unsigned stride) {
  __hmma_m16n16k16_st_c_f16(dst, src, stride, 1);
}

// CIR-LABEL: @store_f32_row(
// LLVM-LABEL: @store_f32_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[SRC:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[STRIDE:.*]] = load i32, ptr %{{.*}}, align 4
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.store.d.row.stride.f32"
// LLVM: %[[S0:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[SP1:.*]] = getelementptr{{.*}} float, ptr %[[SRC]], i{{32|64}} 1
// LLVM: %[[S1:.*]] = load float, ptr %[[SP1]], align 4
// LLVM: %[[SP2:.*]] = getelementptr{{.*}} float, ptr %[[SRC]], i{{32|64}} 2
// LLVM: %[[S2:.*]] = load float, ptr %[[SP2]], align 4
// LLVM: %[[SP3:.*]] = getelementptr{{.*}} float, ptr %[[SRC]], i{{32|64}} 3
// LLVM: %[[S3:.*]] = load float, ptr %[[SP3]], align 4
// LLVM: %[[SP4:.*]] = getelementptr{{.*}} float, ptr %[[SRC]], i{{32|64}} 4
// LLVM: %[[S4:.*]] = load float, ptr %[[SP4]], align 4
// LLVM: %[[SP5:.*]] = getelementptr{{.*}} float, ptr %[[SRC]], i{{32|64}} 5
// LLVM: %[[S5:.*]] = load float, ptr %[[SP5]], align 4
// LLVM: %[[SP6:.*]] = getelementptr{{.*}} float, ptr %[[SRC]], i{{32|64}} 6
// LLVM: %[[S6:.*]] = load float, ptr %[[SP6]], align 4
// LLVM: %[[SP7:.*]] = getelementptr{{.*}} float, ptr %[[SRC]], i{{32|64}} 7
// LLVM: %[[S7:.*]] = load float, ptr %[[SP7]], align 4
// LLVM: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(ptr %[[DST]], float %[[S0]], float %[[S1]], float %[[S2]], float %[[S3]], float %[[S4]], float %[[S5]], float %[[S6]], float %[[S7]], i32 %[[STRIDE]])
// LLVM: ret void
DEVICE void store_f32_row(float *dst, const float *src, unsigned stride) {
  __hmma_m16n16k16_st_c_f32(dst, src, stride, 0);
}

// CIR-LABEL: @store_f32_col(
// LLVM-LABEL: @store_f32_col(
// CIR: cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.store.d.col.stride.f32"
// LLVM: call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p0(ptr %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i32 %{{.*}})
// LLVM: ret void
DEVICE void store_f32_col(float *dst, const float *src, unsigned stride) {
  __hmma_m16n16k16_st_c_f32(dst, src, stride, 1);
}

// CIR-LABEL: @mma_f16_f16_row_row(
// LLVM-LABEL: @mma_f16_f16_row_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[A:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[B:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[C:.*]] = load ptr, ptr %{{.*}}
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f16.f16"
// CIR: cir.extract_member %[[RESULT]][0]
// CIR: cir.extract_member %[[RESULT]][3]
// LLVM: %[[A0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[AR0:.*]] = bitcast i32 %[[A0]] to <2 x half>
// LLVM: %[[AP1:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 1
// LLVM: %[[A1:.*]] = load i32, ptr %[[AP1]], align 4
// LLVM: %[[AR1:.*]] = bitcast i32 %[[A1]] to <2 x half>
// LLVM: %[[AP2:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 2
// LLVM: %[[A2:.*]] = load i32, ptr %[[AP2]], align 4
// LLVM: %[[AR2:.*]] = bitcast i32 %[[A2]] to <2 x half>
// LLVM: %[[AP3:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 3
// LLVM: %[[A3:.*]] = load i32, ptr %[[AP3]], align 4
// LLVM: %[[AR3:.*]] = bitcast i32 %[[A3]] to <2 x half>
// LLVM: %[[AP4:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 4
// LLVM: %[[A4:.*]] = load i32, ptr %[[AP4]], align 4
// LLVM: %[[AR4:.*]] = bitcast i32 %[[A4]] to <2 x half>
// LLVM: %[[AP5:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 5
// LLVM: %[[A5:.*]] = load i32, ptr %[[AP5]], align 4
// LLVM: %[[AR5:.*]] = bitcast i32 %[[A5]] to <2 x half>
// LLVM: %[[AP6:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 6
// LLVM: %[[A6:.*]] = load i32, ptr %[[AP6]], align 4
// LLVM: %[[AR6:.*]] = bitcast i32 %[[A6]] to <2 x half>
// LLVM: %[[AP7:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 7
// LLVM: %[[A7:.*]] = load i32, ptr %[[AP7]], align 4
// LLVM: %[[AR7:.*]] = bitcast i32 %[[A7]] to <2 x half>
// LLVM: %[[B0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[BR0:.*]] = bitcast i32 %[[B0]] to <2 x half>
// LLVM: %[[BP1:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 1
// LLVM: %[[B1:.*]] = load i32, ptr %[[BP1]], align 4
// LLVM: %[[BR1:.*]] = bitcast i32 %[[B1]] to <2 x half>
// LLVM: %[[BP2:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 2
// LLVM: %[[B2:.*]] = load i32, ptr %[[BP2]], align 4
// LLVM: %[[BR2:.*]] = bitcast i32 %[[B2]] to <2 x half>
// LLVM: %[[BP3:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 3
// LLVM: %[[B3:.*]] = load i32, ptr %[[BP3]], align 4
// LLVM: %[[BR3:.*]] = bitcast i32 %[[B3]] to <2 x half>
// LLVM: %[[BP4:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 4
// LLVM: %[[B4:.*]] = load i32, ptr %[[BP4]], align 4
// LLVM: %[[BR4:.*]] = bitcast i32 %[[B4]] to <2 x half>
// LLVM: %[[BP5:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 5
// LLVM: %[[B5:.*]] = load i32, ptr %[[BP5]], align 4
// LLVM: %[[BR5:.*]] = bitcast i32 %[[B5]] to <2 x half>
// LLVM: %[[BP6:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 6
// LLVM: %[[B6:.*]] = load i32, ptr %[[BP6]], align 4
// LLVM: %[[BR6:.*]] = bitcast i32 %[[B6]] to <2 x half>
// LLVM: %[[BP7:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 7
// LLVM: %[[B7:.*]] = load i32, ptr %[[BP7]], align 4
// LLVM: %[[BR7:.*]] = bitcast i32 %[[B7]] to <2 x half>
// LLVM: %[[C0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CR0:.*]] = bitcast i32 %[[C0]] to <2 x half>
// LLVM: %[[CP1:.*]] = getelementptr{{.*}} i32, ptr %[[C]], i{{32|64}} 1
// LLVM: %[[C1:.*]] = load i32, ptr %[[CP1]], align 4
// LLVM: %[[CR1:.*]] = bitcast i32 %[[C1]] to <2 x half>
// LLVM: %[[CP2:.*]] = getelementptr{{.*}} i32, ptr %[[C]], i{{32|64}} 2
// LLVM: %[[C2:.*]] = load i32, ptr %[[CP2]], align 4
// LLVM: %[[CR2:.*]] = bitcast i32 %[[C2]] to <2 x half>
// LLVM: %[[CP3:.*]] = getelementptr{{.*}} i32, ptr %[[C]], i{{32|64}} 3
// LLVM: %[[C3:.*]] = load i32, ptr %[[CP3]], align 4
// LLVM: %[[CR3:.*]] = bitcast i32 %[[C3]] to <2 x half>
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16(<2 x half> %[[AR0]], <2 x half> %[[AR1]], <2 x half> %[[AR2]], <2 x half> %[[AR3]], <2 x half> %[[AR4]], <2 x half> %[[AR5]], <2 x half> %[[AR6]], <2 x half> %[[AR7]], <2 x half> %[[BR0]], <2 x half> %[[BR1]], <2 x half> %[[BR2]], <2 x half> %[[BR3]], <2 x half> %[[BR4]], <2 x half> %[[BR5]], <2 x half> %[[BR6]], <2 x half> %[[BR7]], <2 x half> %[[CR0]], <2 x half> %[[CR1]], <2 x half> %[[CR2]], <2 x half> %[[CR3]])
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
DEVICE void mma_f16_f16_row_row(int *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f16f16(dst, a, b, c, 0, 0);
}

// CIR-LABEL: @mma_f16_f16_row_row_satfinite(
// LLVM-LABEL: @mma_f16_f16_row_row_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f16.f16.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f16_f16_row_row_satfinite(int *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f16f16(dst, a, b, c, 0, 1);
}

// CIR-LABEL: @mma_f16_f16_row_col(
// LLVM-LABEL: @mma_f16_f16_row_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f16.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f16_f16_row_col(int *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f16f16(dst, a, b, c, 1, 0);
}

// CIR-LABEL: @mma_f16_f16_row_col_satfinite(
// LLVM-LABEL: @mma_f16_f16_row_col_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f16.f16.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f16.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f16_f16_row_col_satfinite(int *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f16f16(dst, a, b, c, 1, 1);
}

// CIR-LABEL: @mma_f16_f16_col_row(
// LLVM-LABEL: @mma_f16_f16_col_row(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f16.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f16_f16_col_row(int *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f16f16(dst, a, b, c, 2, 0);
}

// CIR-LABEL: @mma_f16_f16_col_row_satfinite(
// LLVM-LABEL: @mma_f16_f16_col_row_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f16.f16.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f16.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f16_f16_col_row_satfinite(int *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f16f16(dst, a, b, c, 2, 1);
}

// CIR-LABEL: @mma_f16_f16_col_col(
// LLVM-LABEL: @mma_f16_f16_col_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f16.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f16_f16_col_col(int *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f16f16(dst, a, b, c, 3, 0);
}

// CIR-LABEL: @mma_f16_f16_col_col_satfinite(
// LLVM-LABEL: @mma_f16_f16_col_col_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f16.f16.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f16.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f16_f16_col_col_satfinite(int *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f16f16(dst, a, b, c, 3, 1);
}

// CIR-LABEL: @mma_f16_f32_row_row(
// LLVM-LABEL: @mma_f16_f32_row_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[A:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[B:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[C:.*]] = load ptr, ptr %{{.*}}
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f16.f32"
// CIR: cir.extract_member %[[RESULT]][0]
// CIR: cir.extract_member %[[RESULT]][3]
// LLVM: %[[A0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[AR0:.*]] = bitcast i32 %[[A0]] to <2 x half>
// LLVM: %[[AP1:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 1
// LLVM: %[[A1:.*]] = load i32, ptr %[[AP1]], align 4
// LLVM: %[[AR1:.*]] = bitcast i32 %[[A1]] to <2 x half>
// LLVM: %[[AP2:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 2
// LLVM: %[[A2:.*]] = load i32, ptr %[[AP2]], align 4
// LLVM: %[[AR2:.*]] = bitcast i32 %[[A2]] to <2 x half>
// LLVM: %[[AP3:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 3
// LLVM: %[[A3:.*]] = load i32, ptr %[[AP3]], align 4
// LLVM: %[[AR3:.*]] = bitcast i32 %[[A3]] to <2 x half>
// LLVM: %[[AP4:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 4
// LLVM: %[[A4:.*]] = load i32, ptr %[[AP4]], align 4
// LLVM: %[[AR4:.*]] = bitcast i32 %[[A4]] to <2 x half>
// LLVM: %[[AP5:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 5
// LLVM: %[[A5:.*]] = load i32, ptr %[[AP5]], align 4
// LLVM: %[[AR5:.*]] = bitcast i32 %[[A5]] to <2 x half>
// LLVM: %[[AP6:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 6
// LLVM: %[[A6:.*]] = load i32, ptr %[[AP6]], align 4
// LLVM: %[[AR6:.*]] = bitcast i32 %[[A6]] to <2 x half>
// LLVM: %[[AP7:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 7
// LLVM: %[[A7:.*]] = load i32, ptr %[[AP7]], align 4
// LLVM: %[[AR7:.*]] = bitcast i32 %[[A7]] to <2 x half>
// LLVM: %[[B0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[BR0:.*]] = bitcast i32 %[[B0]] to <2 x half>
// LLVM: %[[BP1:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 1
// LLVM: %[[B1:.*]] = load i32, ptr %[[BP1]], align 4
// LLVM: %[[BR1:.*]] = bitcast i32 %[[B1]] to <2 x half>
// LLVM: %[[BP2:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 2
// LLVM: %[[B2:.*]] = load i32, ptr %[[BP2]], align 4
// LLVM: %[[BR2:.*]] = bitcast i32 %[[B2]] to <2 x half>
// LLVM: %[[BP3:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 3
// LLVM: %[[B3:.*]] = load i32, ptr %[[BP3]], align 4
// LLVM: %[[BR3:.*]] = bitcast i32 %[[B3]] to <2 x half>
// LLVM: %[[BP4:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 4
// LLVM: %[[B4:.*]] = load i32, ptr %[[BP4]], align 4
// LLVM: %[[BR4:.*]] = bitcast i32 %[[B4]] to <2 x half>
// LLVM: %[[BP5:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 5
// LLVM: %[[B5:.*]] = load i32, ptr %[[BP5]], align 4
// LLVM: %[[BR5:.*]] = bitcast i32 %[[B5]] to <2 x half>
// LLVM: %[[BP6:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 6
// LLVM: %[[B6:.*]] = load i32, ptr %[[BP6]], align 4
// LLVM: %[[BR6:.*]] = bitcast i32 %[[B6]] to <2 x half>
// LLVM: %[[BP7:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 7
// LLVM: %[[B7:.*]] = load i32, ptr %[[BP7]], align 4
// LLVM: %[[BR7:.*]] = bitcast i32 %[[B7]] to <2 x half>
// LLVM: %[[C0:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[CP1:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 1
// LLVM: %[[C1:.*]] = load float, ptr %[[CP1]], align 4
// LLVM: %[[CP2:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 2
// LLVM: %[[C2:.*]] = load float, ptr %[[CP2]], align 4
// LLVM: %[[CP3:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 3
// LLVM: %[[C3:.*]] = load float, ptr %[[CP3]], align 4
// LLVM: %[[CP4:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 4
// LLVM: %[[C4:.*]] = load float, ptr %[[CP4]], align 4
// LLVM: %[[CP5:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 5
// LLVM: %[[C5:.*]] = load float, ptr %[[CP5]], align 4
// LLVM: %[[CP6:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 6
// LLVM: %[[C6:.*]] = load float, ptr %[[CP6]], align 4
// LLVM: %[[CP7:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 7
// LLVM: %[[C7:.*]] = load float, ptr %[[CP7]], align 4
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f32(<2 x half> %[[AR0]], <2 x half> %[[AR1]], <2 x half> %[[AR2]], <2 x half> %[[AR3]], <2 x half> %[[AR4]], <2 x half> %[[AR5]], <2 x half> %[[AR6]], <2 x half> %[[AR7]], <2 x half> %[[BR0]], <2 x half> %[[BR1]], <2 x half> %[[BR2]], <2 x half> %[[BR3]], <2 x half> %[[BR4]], <2 x half> %[[BR5]], <2 x half> %[[BR6]], <2 x half> %[[BR7]], float %[[C0]], float %[[C1]], float %[[C2]], float %[[C3]], float %[[C4]], float %[[C5]], float %[[C6]], float %[[C7]])
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
DEVICE void mma_f16_f32_row_row(int *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f16f32(dst, a, b, c, 0, 0);
}

// CIR-LABEL: @mma_f16_f32_row_row_satfinite(
// LLVM-LABEL: @mma_f16_f32_row_row_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f16.f32.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f32.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f16_f32_row_row_satfinite(int *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f16f32(dst, a, b, c, 0, 1);
}

// CIR-LABEL: @mma_f16_f32_row_col(
// LLVM-LABEL: @mma_f16_f32_row_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f16.f32"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f32(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f16_f32_row_col(int *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f16f32(dst, a, b, c, 1, 0);
}

// CIR-LABEL: @mma_f16_f32_row_col_satfinite(
// LLVM-LABEL: @mma_f16_f32_row_col_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f16.f32.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f32.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f16_f32_row_col_satfinite(int *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f16f32(dst, a, b, c, 1, 1);
}

// CIR-LABEL: @mma_f16_f32_col_row(
// LLVM-LABEL: @mma_f16_f32_col_row(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f16.f32"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f32(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f16_f32_col_row(int *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f16f32(dst, a, b, c, 2, 0);
}

// CIR-LABEL: @mma_f16_f32_col_row_satfinite(
// LLVM-LABEL: @mma_f16_f32_col_row_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f16.f32.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f32.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f16_f32_col_row_satfinite(int *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f16f32(dst, a, b, c, 2, 1);
}

// CIR-LABEL: @mma_f16_f32_col_col(
// LLVM-LABEL: @mma_f16_f32_col_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f16.f32"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f32(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f16_f32_col_col(int *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f16f32(dst, a, b, c, 3, 0);
}

// CIR-LABEL: @mma_f16_f32_col_col_satfinite(
// LLVM-LABEL: @mma_f16_f32_col_col_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f16.f32.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f32.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f16_f32_col_col_satfinite(int *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f16f32(dst, a, b, c, 3, 1);
}

// CIR-LABEL: @mma_f32_f16_row_row(
// LLVM-LABEL: @mma_f32_f16_row_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[A:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[B:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[C:.*]] = load ptr, ptr %{{.*}}
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f32.f16"
// CIR: cir.extract_member %[[RESULT]][0]
// CIR: cir.extract_member %[[RESULT]][7]
// LLVM: %[[A0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[AR0:.*]] = bitcast i32 %[[A0]] to <2 x half>
// LLVM: %[[AP1:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 1
// LLVM: %[[A1:.*]] = load i32, ptr %[[AP1]], align 4
// LLVM: %[[AR1:.*]] = bitcast i32 %[[A1]] to <2 x half>
// LLVM: %[[AP2:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 2
// LLVM: %[[A2:.*]] = load i32, ptr %[[AP2]], align 4
// LLVM: %[[AR2:.*]] = bitcast i32 %[[A2]] to <2 x half>
// LLVM: %[[AP3:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 3
// LLVM: %[[A3:.*]] = load i32, ptr %[[AP3]], align 4
// LLVM: %[[AR3:.*]] = bitcast i32 %[[A3]] to <2 x half>
// LLVM: %[[AP4:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 4
// LLVM: %[[A4:.*]] = load i32, ptr %[[AP4]], align 4
// LLVM: %[[AR4:.*]] = bitcast i32 %[[A4]] to <2 x half>
// LLVM: %[[AP5:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 5
// LLVM: %[[A5:.*]] = load i32, ptr %[[AP5]], align 4
// LLVM: %[[AR5:.*]] = bitcast i32 %[[A5]] to <2 x half>
// LLVM: %[[AP6:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 6
// LLVM: %[[A6:.*]] = load i32, ptr %[[AP6]], align 4
// LLVM: %[[AR6:.*]] = bitcast i32 %[[A6]] to <2 x half>
// LLVM: %[[AP7:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 7
// LLVM: %[[A7:.*]] = load i32, ptr %[[AP7]], align 4
// LLVM: %[[AR7:.*]] = bitcast i32 %[[A7]] to <2 x half>
// LLVM: %[[B0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[BR0:.*]] = bitcast i32 %[[B0]] to <2 x half>
// LLVM: %[[BP1:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 1
// LLVM: %[[B1:.*]] = load i32, ptr %[[BP1]], align 4
// LLVM: %[[BR1:.*]] = bitcast i32 %[[B1]] to <2 x half>
// LLVM: %[[BP2:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 2
// LLVM: %[[B2:.*]] = load i32, ptr %[[BP2]], align 4
// LLVM: %[[BR2:.*]] = bitcast i32 %[[B2]] to <2 x half>
// LLVM: %[[BP3:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 3
// LLVM: %[[B3:.*]] = load i32, ptr %[[BP3]], align 4
// LLVM: %[[BR3:.*]] = bitcast i32 %[[B3]] to <2 x half>
// LLVM: %[[BP4:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 4
// LLVM: %[[B4:.*]] = load i32, ptr %[[BP4]], align 4
// LLVM: %[[BR4:.*]] = bitcast i32 %[[B4]] to <2 x half>
// LLVM: %[[BP5:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 5
// LLVM: %[[B5:.*]] = load i32, ptr %[[BP5]], align 4
// LLVM: %[[BR5:.*]] = bitcast i32 %[[B5]] to <2 x half>
// LLVM: %[[BP6:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 6
// LLVM: %[[B6:.*]] = load i32, ptr %[[BP6]], align 4
// LLVM: %[[BR6:.*]] = bitcast i32 %[[B6]] to <2 x half>
// LLVM: %[[BP7:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 7
// LLVM: %[[B7:.*]] = load i32, ptr %[[BP7]], align 4
// LLVM: %[[BR7:.*]] = bitcast i32 %[[B7]] to <2 x half>
// LLVM: %[[C0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CR0:.*]] = bitcast i32 %[[C0]] to <2 x half>
// LLVM: %[[CP1:.*]] = getelementptr{{.*}} i32, ptr %[[C]], i{{32|64}} 1
// LLVM: %[[C1:.*]] = load i32, ptr %[[CP1]], align 4
// LLVM: %[[CR1:.*]] = bitcast i32 %[[C1]] to <2 x half>
// LLVM: %[[CP2:.*]] = getelementptr{{.*}} i32, ptr %[[C]], i{{32|64}} 2
// LLVM: %[[C2:.*]] = load i32, ptr %[[CP2]], align 4
// LLVM: %[[CR2:.*]] = bitcast i32 %[[C2]] to <2 x half>
// LLVM: %[[CP3:.*]] = getelementptr{{.*}} i32, ptr %[[C]], i{{32|64}} 3
// LLVM: %[[C3:.*]] = load i32, ptr %[[CP3]], align 4
// LLVM: %[[CR3:.*]] = bitcast i32 %[[C3]] to <2 x half>
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f16(<2 x half> %[[AR0]], <2 x half> %[[AR1]], <2 x half> %[[AR2]], <2 x half> %[[AR3]], <2 x half> %[[AR4]], <2 x half> %[[AR5]], <2 x half> %[[AR6]], <2 x half> %[[AR7]], <2 x half> %[[BR0]], <2 x half> %[[BR1]], <2 x half> %[[BR2]], <2 x half> %[[BR3]], <2 x half> %[[BR4]], <2 x half> %[[BR5]], <2 x half> %[[BR6]], <2 x half> %[[BR7]], <2 x half> %[[CR0]], <2 x half> %[[CR1]], <2 x half> %[[CR2]], <2 x half> %[[CR3]])
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
DEVICE void mma_f32_f16_row_row(float *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f32f16(dst, a, b, c, 0, 0);
}

// CIR-LABEL: @mma_f32_f16_row_row_satfinite(
// LLVM-LABEL: @mma_f32_f16_row_row_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f32.f16.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f16.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f32_f16_row_row_satfinite(float *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f32f16(dst, a, b, c, 0, 1);
}

// CIR-LABEL: @mma_f32_f16_row_col(
// LLVM-LABEL: @mma_f32_f16_row_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f32.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f32_f16_row_col(float *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f32f16(dst, a, b, c, 1, 0);
}

// CIR-LABEL: @mma_f32_f16_row_col_satfinite(
// LLVM-LABEL: @mma_f32_f16_row_col_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f32.f16.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f16.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f32_f16_row_col_satfinite(float *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f32f16(dst, a, b, c, 1, 1);
}

// CIR-LABEL: @mma_f32_f16_col_row(
// LLVM-LABEL: @mma_f32_f16_col_row(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f32.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f32_f16_col_row(float *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f32f16(dst, a, b, c, 2, 0);
}

// CIR-LABEL: @mma_f32_f16_col_row_satfinite(
// LLVM-LABEL: @mma_f32_f16_col_row_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f32.f16.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f16.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f32_f16_col_row_satfinite(float *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f32f16(dst, a, b, c, 2, 1);
}

// CIR-LABEL: @mma_f32_f16_col_col(
// LLVM-LABEL: @mma_f32_f16_col_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f32.f16"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f32_f16_col_col(float *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f32f16(dst, a, b, c, 3, 0);
}

// CIR-LABEL: @mma_f32_f16_col_col_satfinite(
// LLVM-LABEL: @mma_f32_f16_col_col_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f32.f16.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f16.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
DEVICE void mma_f32_f16_col_col_satfinite(float *dst, const int *a, const int *b, const int *c) {
  __hmma_m16n16k16_mma_f32f16(dst, a, b, c, 3, 1);
}

// CIR-LABEL: @mma_f32_f32_row_row(
// LLVM-LABEL: @mma_f32_f32_row_row(
// LLVM: %[[DST:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[A:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[B:.*]] = load ptr, ptr %{{.*}}
// LLVM: %[[C:.*]] = load ptr, ptr %{{.*}}
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f32.f32"
// CIR: cir.extract_member %[[RESULT]][0]
// CIR: cir.extract_member %[[RESULT]][7]
// LLVM: %[[A0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[AR0:.*]] = bitcast i32 %[[A0]] to <2 x half>
// LLVM: %[[AP1:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 1
// LLVM: %[[A1:.*]] = load i32, ptr %[[AP1]], align 4
// LLVM: %[[AR1:.*]] = bitcast i32 %[[A1]] to <2 x half>
// LLVM: %[[AP2:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 2
// LLVM: %[[A2:.*]] = load i32, ptr %[[AP2]], align 4
// LLVM: %[[AR2:.*]] = bitcast i32 %[[A2]] to <2 x half>
// LLVM: %[[AP3:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 3
// LLVM: %[[A3:.*]] = load i32, ptr %[[AP3]], align 4
// LLVM: %[[AR3:.*]] = bitcast i32 %[[A3]] to <2 x half>
// LLVM: %[[AP4:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 4
// LLVM: %[[A4:.*]] = load i32, ptr %[[AP4]], align 4
// LLVM: %[[AR4:.*]] = bitcast i32 %[[A4]] to <2 x half>
// LLVM: %[[AP5:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 5
// LLVM: %[[A5:.*]] = load i32, ptr %[[AP5]], align 4
// LLVM: %[[AR5:.*]] = bitcast i32 %[[A5]] to <2 x half>
// LLVM: %[[AP6:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 6
// LLVM: %[[A6:.*]] = load i32, ptr %[[AP6]], align 4
// LLVM: %[[AR6:.*]] = bitcast i32 %[[A6]] to <2 x half>
// LLVM: %[[AP7:.*]] = getelementptr{{.*}} i32, ptr %[[A]], i{{32|64}} 7
// LLVM: %[[A7:.*]] = load i32, ptr %[[AP7]], align 4
// LLVM: %[[AR7:.*]] = bitcast i32 %[[A7]] to <2 x half>
// LLVM: %[[B0:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[BR0:.*]] = bitcast i32 %[[B0]] to <2 x half>
// LLVM: %[[BP1:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 1
// LLVM: %[[B1:.*]] = load i32, ptr %[[BP1]], align 4
// LLVM: %[[BR1:.*]] = bitcast i32 %[[B1]] to <2 x half>
// LLVM: %[[BP2:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 2
// LLVM: %[[B2:.*]] = load i32, ptr %[[BP2]], align 4
// LLVM: %[[BR2:.*]] = bitcast i32 %[[B2]] to <2 x half>
// LLVM: %[[BP3:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 3
// LLVM: %[[B3:.*]] = load i32, ptr %[[BP3]], align 4
// LLVM: %[[BR3:.*]] = bitcast i32 %[[B3]] to <2 x half>
// LLVM: %[[BP4:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 4
// LLVM: %[[B4:.*]] = load i32, ptr %[[BP4]], align 4
// LLVM: %[[BR4:.*]] = bitcast i32 %[[B4]] to <2 x half>
// LLVM: %[[BP5:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 5
// LLVM: %[[B5:.*]] = load i32, ptr %[[BP5]], align 4
// LLVM: %[[BR5:.*]] = bitcast i32 %[[B5]] to <2 x half>
// LLVM: %[[BP6:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 6
// LLVM: %[[B6:.*]] = load i32, ptr %[[BP6]], align 4
// LLVM: %[[BR6:.*]] = bitcast i32 %[[B6]] to <2 x half>
// LLVM: %[[BP7:.*]] = getelementptr{{.*}} i32, ptr %[[B]], i{{32|64}} 7
// LLVM: %[[B7:.*]] = load i32, ptr %[[BP7]], align 4
// LLVM: %[[BR7:.*]] = bitcast i32 %[[B7]] to <2 x half>
// LLVM: %[[C0:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[CP1:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 1
// LLVM: %[[C1:.*]] = load float, ptr %[[CP1]], align 4
// LLVM: %[[CP2:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 2
// LLVM: %[[C2:.*]] = load float, ptr %[[CP2]], align 4
// LLVM: %[[CP3:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 3
// LLVM: %[[C3:.*]] = load float, ptr %[[CP3]], align 4
// LLVM: %[[CP4:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 4
// LLVM: %[[C4:.*]] = load float, ptr %[[CP4]], align 4
// LLVM: %[[CP5:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 5
// LLVM: %[[C5:.*]] = load float, ptr %[[CP5]], align 4
// LLVM: %[[CP6:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 6
// LLVM: %[[C6:.*]] = load float, ptr %[[CP6]], align 4
// LLVM: %[[CP7:.*]] = getelementptr{{.*}} float, ptr %[[C]], i{{32|64}} 7
// LLVM: %[[C7:.*]] = load float, ptr %[[CP7]], align 4
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> %[[AR0]], <2 x half> %[[AR1]], <2 x half> %[[AR2]], <2 x half> %[[AR3]], <2 x half> %[[AR4]], <2 x half> %[[AR5]], <2 x half> %[[AR6]], <2 x half> %[[AR7]], <2 x half> %[[BR0]], <2 x half> %[[BR1]], <2 x half> %[[BR2]], <2 x half> %[[BR3]], <2 x half> %[[BR4]], <2 x half> %[[BR5]], <2 x half> %[[BR6]], <2 x half> %[[BR7]], float %[[C0]], float %[[C1]], float %[[C2]], float %[[C3]], float %[[C4]], float %[[C5]], float %[[C6]], float %[[C7]])
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
DEVICE void mma_f32_f32_row_row(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 0, 0);
}

// CIR-LABEL: @mma_f32_f32_row_row_satfinite(
// LLVM-LABEL: @mma_f32_f32_row_row_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.row.f32.f32.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f32_f32_row_row_satfinite(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 0, 1);
}

// CIR-LABEL: @mma_f32_f32_row_col(
// LLVM-LABEL: @mma_f32_f32_row_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f32.f32"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f32_f32_row_col(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 1, 0);
}

// CIR-LABEL: @mma_f32_f32_row_col_satfinite(
// LLVM-LABEL: @mma_f32_f32_row_col_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.row.col.f32.f32.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f32_f32_row_col_satfinite(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 1, 1);
}

// CIR-LABEL: @mma_f32_f32_col_row(
// LLVM-LABEL: @mma_f32_f32_col_row(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f32.f32"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f32(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f32_f32_col_row(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 2, 0);
}

// CIR-LABEL: @mma_f32_f32_col_row_satfinite(
// LLVM-LABEL: @mma_f32_f32_col_row_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.row.f32.f32.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f32.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f32_f32_col_row_satfinite(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 2, 1);
}

// CIR-LABEL: @mma_f32_f32_col_col(
// LLVM-LABEL: @mma_f32_f32_col_col(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f32.f32"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f32_f32_col_col(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 3, 0);
}

// CIR-LABEL: @mma_f32_f32_col_col_satfinite(
// LLVM-LABEL: @mma_f32_f32_col_col_satfinite(
// CIR: %[[RESULT:.*]] = cir.call_llvm_intrinsic "nvvm.wmma.m16n16k16.mma.col.col.f32.f32.satfinite"
// LLVM: %[[RESULT:.*]] = call{{.*}} { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32.satfinite(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
DEVICE void mma_f32_f32_col_col_satfinite(float *dst, const int *a, const int *b, const float *c) {
  __hmma_m16n16k16_mma_f32f32(dst, a, b, c, 3, 1);
}

