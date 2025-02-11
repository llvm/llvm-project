// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

int test_float_isinf_sign(float x) {
    // CIR-LABEL: test_float_isinf_sign
    // CIR: %[[TMP0:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.float>, !cir.float
    // CIR: %[[TMP1:.*]] = cir.fabs %[[TMP0]] : !cir.float
    // CIR: %[[IS_INF:.*]] = cir.is_fp_class %[[TMP1]], 516 : (!cir.float) -> !cir.bool
    // CIR: %[[IS_NEG:.*]] = cir.signbit %[[TMP0]] : !cir.float -> !cir.bool
    // CIR: %[[C_0:.*]] = cir.const #cir.int<0> : !s32i
    // CIR: %[[C_1:.*]] = cir.const #cir.int<1> : !s32i
    // CIR: %[[C_m1:.*]] = cir.const #cir.int<-1> : !s32i
    // CIR: %[[TMP4:.*]] = cir.select if %[[IS_NEG]] then %[[C_m1]] else %[[C_1]] : (!cir.bool, !s32i, !s32i) -> !s32i
    // CIR: %[[RET:.*]] = cir.select if %[[IS_INF]] then %[[TMP4]] else %[[C_0]] : (!cir.bool, !s32i, !s32i) -> !s32i
    // CIR: cir.store %[[RET]], %{{.*}} : !s32i, !cir.ptr<!s32i>

    // LLVM-LABEL: test_float_isinf_sign
    // LLVM: %[[TMP0:.*]] = load float, ptr %{{.*}}
    // LLVM: %[[TMP1:.*]] = call float @llvm.fabs.f32(float %[[TMP0]])
    // LLVM: %[[IS_INF:.*]] = call i1 @llvm.is.fpclass.f32(float %[[TMP1]], i32 516)
    // LLVM: %[[TMP1:.*]] = bitcast float %[[TMP0]] to i32
    // LLVM: %[[IS_NEG:.*]] = icmp slt i32 %[[TMP1]], 0
    // LLVM: %[[TMP2:.*]] = select i1 %[[IS_NEG]], i32 -1, i32 1
    // LLVM: %[[RET:.*]] = select i1 %[[IS_INF]], i32 %[[TMP2]], i32 0
    // LLVM: store i32 %[[RET]], ptr %{{.*}}, align 4
    return __builtin_isinf_sign(x);
}
