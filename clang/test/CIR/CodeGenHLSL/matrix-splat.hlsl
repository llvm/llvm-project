// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void constant_int_splat() {
    int4x4 a = 1;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.matrix<4 x 4 x !s32i>>
// CIR: %[[CONST_VALUE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[CONST_VALUE]] : !s32i, !cir.matrix<4 x 4 x !s32i>
// CIR: cir.store {{.*}} %[[SPLAT]], %[[A_ADDR]] : !cir.matrix<4 x 4 x !s32i>, !cir.ptr<!cir.matrix<4 x 4 x !s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca [4 x <4 x i32>], align 4
// LLVM: store <16 x i32> splat (i32 1), ptr %[[A_ADDR]], align 4

void constant_fp_splat() {
    float2x2 a = 3.25;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.matrix<2 x 2 x !cir.float>>
// CIR: %[[CONST_VALUE:.*]] = cir.const #cir.fp<3.250000e+00> : !cir.float
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[CONST_VALUE]] : !cir.float, !cir.matrix<2 x 2 x !cir.float>
// CIR: cir.store {{.*}} %[[SPLAT]], %[[A_ADDR]] : !cir.matrix<2 x 2 x !cir.float>, !cir.ptr<!cir.matrix<2 x 2 x !cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca [2 x <2 x float>], align 4
// LLVM: store <4 x float> splat (float 3.250000e+00), ptr %[[A_ADDR]], align 4

void dynamic_fp_splat(float v) {
    float3x3 a = v;
}

// CIR: %[[V_ADDR:.*]] = cir.alloca "v" {{.*}} init : !cir.ptr<!cir.float> loc(#loc21)
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>> loc(#loc22)
// CIR: cir.store %[[V:.*]], %[[V_ADDR]] : !cir.float, !cir.ptr<!cir.float> loc(#loc15)
// CIR: %[[TMP_V:.*]] = cir.load {{.*}} %[[V_ADDR]] : !cir.ptr<!cir.float>, !cir.float loc(#loc14)
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[TMP_V]] : !cir.float, !cir.matrix<3 x 3 x !cir.float> loc(#loc14)
// CIR: cir.store {{.*}} %[[SPLAT]], %[[A_ADDR]] : !cir.matrix<3 x 3 x !cir.float>, !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>

 // LLVM: %[[V_ADDR:.*]] = alloca float, align 4
 // LLVM: %[[A_ADDR:.*]] = alloca [3 x <3 x float>], align 4
 // LLVM: store float %[[V:.*]], ptr %[[V_ADDR]], align 4
 // LLVM: %[[TMP_V:.*]] = load float, ptr %[[V_ADDR]], align 4
 // LLVM: %[[SPLAT_INSERT:.*]] = insertelement <9 x float> poison, float %[[TMP_V]], i64 0
 // LLVM: %[[SPLAT:.*]] = shufflevector <9 x float> %[[SPLAT_INSERT]], <9 x float> poison, <9 x i32> zeroinitializer
 // LLVM: store <9 x float> %[[SPLAT]], ptr %[[A_ADDR]], align 4
