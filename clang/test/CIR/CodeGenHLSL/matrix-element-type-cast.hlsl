// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -x hlsl -std=hlsl2021 -triple spirv-unknown-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void fp_matrix_to_int_matrix() {
    float3x2 a;
    int3x2 b = (int3x2) a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 2 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.matrix<3 x 2 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<3 x 2 x !cir.float>>, !cir.matrix<3 x 2 x !cir.float>
// CIR: %[[RESULT:.*]] = cir.cast float_to_int %[[TMP_A]] : !cir.matrix<3 x 2 x !cir.float> -> !cir.matrix<3 x 2 x !s32i>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.matrix<3 x 2 x !s32i>, !cir.ptr<!cir.matrix<3 x 2 x !s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca [2 x <3 x float>], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [2 x <3 x i32>], align 4
// LLVM: %[[TMP_A:.*]] = load <6 x float>, ptr %[[A_ADDR]], align 4
// LLVM: %[[RESULT:.*]] = fptosi <6 x float> %[[TMP_A]] to <6 x i32>
// LLVM: store <6 x i32> %[[RESULT]], ptr %[[B_ADDR]], align 4

void int_matrix_to_fp_matrix() {
    int3x2 a;
    float3x2 b = (float3x2) a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 2 x !s32i>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.matrix<3 x 2 x !cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<3 x 2 x !s32i>>, !cir.matrix<3 x 2 x !s32i>
// CIR: %[[RESULT:.*]] = cir.cast int_to_float %[[TMP_A]] : !cir.matrix<3 x 2 x !s32i> -> !cir.matrix<3 x 2 x !cir.float>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.matrix<3 x 2 x !cir.float>, !cir.ptr<!cir.matrix<3 x 2 x !cir.float>>

// TODO(CIR): Update sitofp check once fast math flags are implemented

// LLVM: %[[A_ADDR:.*]] = alloca [2 x <3 x i32>], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [2 x <3 x float>], align 4
// LLVM: %[[TMP_A:.*]] = load <6 x i32>, ptr %[[A_ADDR]], align 4
// LLVM: %[[RESULT:.*]] = sitofp{{.*}} <6 x i32> %[[TMP_A]] to <6 x float>
// LLVM: store <6 x float> %[[RESULT]], ptr %[[B_ADDR]], align 4

void fp_matrix_to_double_matrix() {
    float3x2 a;
    double3x2 b = (double3x2) a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 2 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.matrix<3 x 2 x !cir.double>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<3 x 2 x !cir.float>>, !cir.matrix<3 x 2 x !cir.float>
// CIR: %[[RESULT:.*]] = cir.cast floating %[[TMP_A]] : !cir.matrix<3 x 2 x !cir.float> -> !cir.matrix<3 x 2 x !cir.double>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.matrix<3 x 2 x !cir.double>, !cir.ptr<!cir.matrix<3 x 2 x !cir.double>>

// TODO(CIR): Update sitofp check once fast math flags are implemented

// LLVM: %[[A_ADDR:.*]] = alloca [2 x <3 x float>], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [2 x <3 x double>], align 8
// LLVM: %[[TMP_A:.*]] = load <6 x float>, ptr %[[A_ADDR]], align 4
// LLVM: %[[RESULT:.*]] = fpext{{.*}} <6 x float> %[[TMP_A]] to <6 x double>
// LLVM: store <6 x double> %[[RESULT]], ptr %[[B_ADDR]], align 8

void int_matrix_to_int64_matrix() {
    int3x2 a;
    int64_t3x2 b = (int64_t3x2) a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 2 x !s32i>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.matrix<3 x 2 x !s64i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<3 x 2 x !s32i>>, !cir.matrix<3 x 2 x !s32i>
// CIR: %[[RESULT:.*]] = cir.cast integral %[[TMP_A]] : !cir.matrix<3 x 2 x !s32i> -> !cir.matrix<3 x 2 x !s64i>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.matrix<3 x 2 x !s64i>, !cir.ptr<!cir.matrix<3 x 2 x !s64i>>

// LLVM: %[[A_ADDR:.*]] = alloca [2 x <3 x i32>], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [2 x <3 x i64>], align 8
// LLVM: %[[TMP_A:.*]] = load <6 x i32>, ptr %[[A_ADDR]], align 4
// LLVM: %[[RESULT:.*]] = sext <6 x i32> %[[TMP_A]] to <6 x i64>
// LLVM: store <6 x i64> %[[RESULT]], ptr %[[B_ADDR]], align 8
