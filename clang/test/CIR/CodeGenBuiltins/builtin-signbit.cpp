// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void test_signbit_float(float val) {
    // CIR-LABEL: test_signbit_float
    // CIR: %{{.+}} = cir.signbit %{{.+}} : !cir.float -> !s32i

    // LLVM-LABEL: test_signbit_float
    // LLVM: [[TMP1:%.*]] = bitcast float %{{.+}} to i32
    // LLVM: [[TMP2:%.*]] = icmp slt i32 [[TMP1]], 0
    // LLVM: %{{.+}} = zext i1 [[TMP2]] to i32

    // OGCG-LABEL: test_signbit_float
    // OGCG: [[TMP1:%.*]] = bitcast float %{{.+}} to i32
    // OGCG: [[TMP2:%.*]] = icmp slt i32 [[TMP1]], 0
    // OGCG: %{{.+}} = zext i1 [[TMP2]] to i32
    __builtin_signbit(val);
}

void test_signbit_double(double val) {
    // CIR-LABEL: test_signbit_double
    // CIR: %{{.+}} = cir.signbit %{{.+}} : !cir.float -> !s32i

    // LLVM-LABEL: test_signbit_double
    // LLVM: [[CONV:%.*]] = fptrunc double %{{.+}} to float
    // LLVM: [[TMP1:%.*]] = bitcast float [[CONV]] to i32
    // LLVM: [[TMP2:%.*]] = icmp slt i32 [[TMP1]], 0
    // LLVM: %{{.+}} = zext i1 [[TMP2]] to i32
    
    // OGCG-LABEL: test_signbit_double
    // OGCG: [[CONV:%.*]] = fptrunc double %{{.+}} to float
    // OGCG: [[TMP1:%.*]] = bitcast float [[CONV]] to i32
    // OGCG: [[TMP2:%.*]] = icmp slt i32 [[TMP1]], 0
    // OGCG: %{{.+}} = zext i1 [[TMP2]] to i32
    __builtin_signbitf(val);
}