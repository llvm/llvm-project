// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

float num;
float _Complex a = {num, num};

// CIR-BEFORE-LPP: cir.global external @num = #cir.fp<0.000000e+00> : !cir.float
// CIR-BEFORE-LPP: cir.global external @a = ctor : !cir.complex<!cir.float> {
// CIR-BEFORE-LPP:  %[[THIS:.*]] = cir.get_global @a : !cir.ptr<!cir.complex<!cir.float>>
// CIR-BEFORE-LPP:  %[[NUM:.*]] = cir.get_global @num : !cir.ptr<!cir.float>
// CIR-BEFORE-LPP:  %[[REAL:.*]] = cir.load{{.*}} %[[NUM]] : !cir.ptr<!cir.float>, !cir.float
// CIR-BEFORE-LPP:  %[[NUM:.*]] = cir.get_global @num : !cir.ptr<!cir.float>
// CIR-BEFORE-LPP:  %[[IMAG:.*]] = cir.load{{.*}} %[[NUM]] : !cir.ptr<!cir.float>, !cir.float
// CIR-BEFORE-LPP:  %[[COMPLEX_VAL:.*]] = cir.complex.create %[[REAL]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-BEFORE-LPP:  cir.store{{.*}} %[[COMPLEX_VAL:.*]], %[[THIS]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-BEFORE-LPP: }

// CIR:  cir.global external @num = #cir.fp<0.000000e+00> : !cir.float
// CIR:  cir.global external @a = #cir.zero : !cir.complex<!cir.float>
// CIR:  cir.func internal private @__cxx_global_var_init()
// CIR:   %[[A_ADDR:.*]] = cir.get_global @a : !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[NUM:.*]] = cir.get_global @num : !cir.ptr<!cir.float>
// CIR:   %[[REAL:.*]] = cir.load{{.*}} %[[NUM]] : !cir.ptr<!cir.float>, !cir.float
// CIR:   %[[NUM:.*]] = cir.get_global @num : !cir.ptr<!cir.float>
// CIR:   %[[IMAG:.*]] = cir.load{{.*}} %[[NUM]] : !cir.ptr<!cir.float>, !cir.float
// CIR:   %[[COMPLEX_VAL:.*]] = cir.complex.create %[[REAL]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR:   cir.store{{.*}} %[[COMPLEX_VAL]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   %[[REAL:.*]] = load float, ptr @num, align 4
// LLVM:   %[[IMAG:.*]] = load float, ptr @num, align 4
// LLVM:   %[[TMP_COMPLEX_VAL:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL]], 0
// LLVM:   %[[COMPLEX_VAL:.*]] = insertvalue { float, float } %[[TMP_COMPLEX_VAL]], float %[[IMAG]], 1
// LLVM:   store { float, float } %[[COMPLEX_VAL]], ptr @a, align 4

// OGCG: define internal void @__cxx_global_var_init() {{.*}} section ".text.startup"
// OGCG:   %[[REAL:.*]] = load float, ptr @num, align 4
// OGCG:   %[[IMAG:.*]] = load float, ptr @num, align 4
// OGCG:   store float %[[REAL]], ptr @a, align 4
// OGCG:   store float %[[IMAG]], ptr getelementptr inbounds nuw ({ float, float }, ptr @a, i32 0, i32 1), align 4
