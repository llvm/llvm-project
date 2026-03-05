// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void test_signbit_float(float val) {
    // CIR-LABEL: test_signbit_float
    // CIR: %{{.+}} = cir.signbit %{{.+}} : !cir.float -> !cir.bool
    // LLVM-LABEL: test_signbit_float
    // LLVM: [[TMP1:%.*]] = bitcast float %{{.+}} to i32
    // LLVM: [[TMP2:%.*]] = icmp slt i32 [[TMP1]], 0
    if (__builtin_signbit(val)) {};
}

void test_signbit_double(double val) {
    // CIR-LABEL: test_signbit_double
    // CIR: %{{.+}} = cir.signbit %{{.+}} : !cir.float -> !cir.bool
    // LLVM-LABEL: test_signbit_double
    // LLVM: [[CONV:%.*]] = fptrunc double %{{.+}} to float
    // LLVM: [[TMP1:%.*]] = bitcast float [[CONV]] to i32
    // LLVM: [[TMP2:%.*]] = icmp slt i32 [[TMP1]], 0
    if (__builtin_signbitf(val)) {}
}

void test_signbit_long_double(long double val) {
    // CIR: test_signbit_long_double
    // LLVM: test_signbit_long_double
    if (__builtin_signbitl(val)) {}
    // CIR: %{{.+}} = cir.signbit %{{.+}} : !cir.long_double<!cir.f80> -> !cir.bool
    // LLVM: [[TMP1:%.*]] = bitcast x86_fp80 %{{.+}} to i80
    // LLVM: [[TMP2:%.*]] = icmp slt i80 [[TMP1]], 0
}
