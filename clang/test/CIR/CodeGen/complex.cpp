// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int _Complex ci;

float _Complex cf;

int _Complex ci2 = { 1, 2 };

float _Complex cf2 = { 1.0f, 2.0f };

// CIR: cir.global external @ci = #cir.zero : !cir.complex<!s32i>
// CIR: cir.global external @cf = #cir.zero : !cir.complex<!cir.float>
// CIR: cir.global external @ci2 = #cir.const_complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>
// CIR: cir.global external @cf2 = #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>

// LLVM: {{.*}} = global { i32, i32 } zeroinitializer, align 4
// LLVM: {{.*}} = global { float, float } zeroinitializer, align 4
// LLVM: {{.*}} = global { i32, i32 } { i32 1, i32 2 }, align 4
// LLVM: {{.*}} = global { float, float } { float 1.000000e+00, float 2.000000e+00 }, align 4

// OGCG: {{.*}} = global { i32, i32 } zeroinitializer, align 4
// OGCG: {{.*}} = global { float, float } zeroinitializer, align 4
// OGCG: {{.*}} = global { i32, i32 } { i32 1, i32 2 }, align 4
// OGCG: {{.*}} = global { float, float } { float 1.000000e+00, float 2.000000e+00 }, align 4

void foo() { int _Complex c = {}; }

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[INIT]], align 4

// OGCG: %[[COMPLEX:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store i32 0, ptr %[[C_REAL_PTR]], align 4
// OGCG: store i32 0, ptr %[[C_IMAG_PTR]], align 4

void foo2() { int _Complex c = {1, 2}; }

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.const_complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } { i32 1, i32 2 }, ptr %[[INIT]], align 4

// OGCG: %[[COMPLEX:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store i32 1, ptr %[[C_REAL_PTR]], align 4
// OGCG: store i32 2, ptr %[[C_IMAG_PTR]], align 4

void foo3() {
  int a;
  int b;
  int _Complex c = {a, b};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s32i>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s32i>
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[TMP_A]], %[[TMP_B]] : !s32i -> !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[TMP_B:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[TMP:.*]] = insertvalue { i32, i32 } undef, i32 %[[TMP_A]], 0
// LLVM: %[[TMP_2:.*]] = insertvalue { i32, i32 } %[[TMP]], i32 %[[TMP_B]], 1
// LLVM: store { i32, i32 } %[[TMP_2]], ptr %[[INIT]], align 4

// OGCG: %[[COMPLEX:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[REAL_VAL:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: %[[IMAG_VAL:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store i32 %[[REAL_VAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG: store i32 %[[IMAG_VAL]], ptr %[[C_IMAG_PTR]], align 4

void foo4() {
  int a;
  int _Complex c = {1, a};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[CONST_1]], %[[TMP_A]] : !s32i -> !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[COMPLEX:.*]] = insertvalue { i32, i32 } { i32 1, i32 undef }, i32 %[[TMP_A]], 1
// LLVM: store { i32, i32 } %[[COMPLEX]], ptr %[[INIT]], align 4

// OGCG: %[[COMPLEX:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[TMP_A:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store i32 1, ptr %[[C_REAL_PTR]], align 4
// OGCG: store i32 %[[TMP_A]], ptr %[[C_IMAG_PTR]], align 4

void foo5() {
  float _Complex c = {1.0f, 2.0f};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[INIT:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: store { float, float } { float 1.000000e+00, float 2.000000e+00 }, ptr %[[INIT]], align 4

// OGCG: %[[COMPLEX]] = alloca { float, float }, align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store float 1.000000e+00, ptr %[[C_REAL_PTR]], align 4
// OGCG: store float 2.000000e+00, ptr %[[C_IMAG_PTR]], align 4

void foo6() {
  float a;
  float b;
  float _Complex c = {a, b};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[TMP_A]], %[[TMP_B]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[COMPLEX:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load float, ptr {{.*}}, align 4
// LLVM: %[[TMP_B:.*]] = load float, ptr {{.*}}, align 4
// LLVM: %[[TMP:.*]] = insertvalue { float, float } undef, float %[[TMP_A]], 0
// LLVM: %[[TMP_2:.*]] = insertvalue { float, float } %[[TMP]], float %[[TMP_B]], 1
// LLVM: store { float, float } %[[TMP_2]], ptr %[[COMPLEX]], align 4

// OGCG: %[[COMPLEX]] = alloca { float, float }, align 4
// OGCG: %[[TMP_A:.*]] = load float, ptr {{.*}}, align 4
// OGCG: %[[TMP_B:.*]] = load float, ptr {{.*}}, align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store float %[[TMP_A]], ptr %[[C_REAL_PTR]], align 4
// OGCG: store float %[[TMP_B]], ptr %[[C_IMAG_PTR]], align 4

void foo7() {
  float a;
  float _Complex c = {a, 2.0f};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CONST_2F:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.float
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[TMP_A]], %[[CONST_2F]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[COMPLEX:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load float, ptr {{.*}}, align 4
// LLVM: %[[TMP:.*]] = insertvalue { float, float } undef, float %[[TMP_A]], 0
// LLVM: %[[TMP_2:.*]] = insertvalue { float, float } %[[TMP]], float 2.000000e+00, 1
// LLVM: store { float, float } %[[TMP_2]], ptr %[[COMPLEX]], align 4

// OGCG: %[[COMPLEX:.*]] = alloca { float, float }, align 4
// OGCG: %[[TMP_A:.*]] = load float, ptr {{.*}}, align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store float %[[TMP_A]], ptr %[[C_REAL_PTR]], align 4
// OGCG: store float 2.000000e+00, ptr %[[C_IMAG_PTR]], align 4

void foo8() {
  double _Complex c = 2.00i;
}

// CIR: %[[COMPLEX:.*]] = cir.const #cir.const_complex<#cir.fp<0.000000e+00> : !cir.double, #cir.fp<2.000000e+00> : !cir.double> : !cir.complex<!cir.double>

// LLVM: %[[COMPLEX:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: store { double, double } { double 0.000000e+00, double 2.000000e+00 }, ptr %[[COMPLEX]], align 8

// OGCG: %[[COMPLEX:.*]] = alloca { double, double }, align 8
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store double 0.000000e+00, ptr %[[C_REAL_PTR]], align 8
// OGCG: store double 2.000000e+00, ptr %[[C_IMAG_PTR]], align 8

void foo9(double a, double b) {
  double _Complex c = __builtin_complex(a, b);
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>, ["c", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[TMP_A]], %[[TMP_B]] : !cir.double -> !cir.complex<!cir.double>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>

// LLVM: %[[COMPLEX:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[TMP_A:.*]] = load double, ptr {{.*}}, align 8
// LLVM: %[[TMP_B:.*]] = load double, ptr {{.*}}, align 8
// LLVM: %[[TMP:.*]] = insertvalue { double, double } undef, double %[[TMP_A]], 0
// LLVM: %[[TMP_2:.*]] = insertvalue { double, double } %[[TMP]], double %[[TMP_B]], 1
// LLVM: store { double, double } %[[TMP_2]], ptr %[[COMPLEX]], align 8

// OGCG: %[[COMPLEX]] = alloca { double, double }, align 8
// OGCG: %[[TMP_A:.*]] = load double, ptr {{.*}}, align 8
// OGCG: %[[TMP_B:.*]] = load double, ptr {{.*}}, align 8
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store double %[[TMP_A]], ptr %[[C_REAL_PTR]], align 8
// OGCG: store double %[[TMP_B]], ptr %[[C_IMAG_PTR]], align 8

void foo10() {
  double _Complex c;
  double *realPtr = &__real__ c;
}

// CIR: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>, ["c"]
// CIR: %[[REAL_PTR:.*]] = cir.complex.real_ptr %[[COMPLEX]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>

// LLVM: %[[COMPLEX:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 0

// OGCG: %[[COMPLEX:.*]] = alloca { double, double }, align 8
// OGCG: %[[REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 0

void foo11() {
  double _Complex c;
  double *imagPtr = &__imag__ c;
}

// CIR: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>, ["c"]
// CIR: %[[IMAG_PTR:.*]] = cir.complex.imag_ptr %[[COMPLEX]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>

// LLVM: %[[COMPLEX:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 1

// OGCG: %[[COMPLEX:.*]] = alloca { double, double }, align 8
// OGCG: %[[IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 1

void foo12() {
  double _Complex c;
  double imag = __imag__ c;
}

// CIR: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>, ["c"]
// CIR: %[[INIT:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["imag", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.double> -> !cir.double
// CIR: cir.store{{.*}} %[[IMAG]], %[[INIT]] : !cir.double, !cir.ptr<!cir.double>

// LLVM: %[[COMPLEX:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[INIT:.*]] = alloca double, i64 1, align 8
// LLVM: %[[TMP:.*]] = load { double, double }, ptr %[[COMPLEX]], align 8
// LLVM: %[[IMAG:.*]] = extractvalue { double, double } %[[TMP]], 1
// LLVM: store double %[[IMAG]], ptr %[[INIT]], align 8

// OGCG: %[[COMPLEX:.*]] = alloca { double, double }, align 8
// OGCG: %[[INIT:.*]] = alloca double, align 8
// OGCG: %[[IMAG:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: %[[TMP:.*]] = load double, ptr %[[IMAG]], align 8
// OGCG: store double %[[TMP]], ptr %[[INIT]], align 8

void foo13() {
  double _Complex c;
  double real = __real__ c;
}

// CIR: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>, ["c"]
// CIR: %[[INIT:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["real", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.double> -> !cir.double
// CIR: cir.store{{.*}} %[[REAL]], %[[INIT]] : !cir.double, !cir.ptr<!cir.double>

// LLVM: %[[COMPLEX:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[INIT:.*]] = alloca double, i64 1, align 8
// LLVM: %[[TMP:.*]] = load { double, double }, ptr %[[COMPLEX]], align 8
// LLVM: %[[REAL:.*]] = extractvalue { double, double } %[[TMP]], 0
// LLVM: store double %[[REAL]], ptr %[[INIT]], align 8

// OGCG: %[[COMPLEX:.*]] = alloca { double, double }, align 8
// OGCG: %[[INIT:.*]] = alloca double, align 8
// OGCG: %[[REAL:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[TMP:.*]] = load double, ptr %[[REAL]], align 8
// OGCG: store double %[[TMP]], ptr %[[INIT]], align 8


void foo14() {
  int _Complex c = 2i;
}

// CIR: %[[COMPLEX:.*]] = cir.const #cir.const_complex<#cir.int<0> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>

// LLVM: %[[COMPLEX:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } { i32 0, i32 2 }, ptr %[[COMPLEX]], align 4

// OGCG: %[[COMPLEX:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: store i32 0, ptr %[[C_REAL_PTR]], align 4
// OGCG: store i32 2, ptr %[[C_IMAG_PTR]], align 4

void foo15() {
  int _Complex a;
  int _Complex b = a;
}

// CIR: %[[COMPLEX_A:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[COMPLEX_B:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[COMPLEX_A]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP_A]], %[[COMPLEX_B]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[COMPLEX_A]], align 4
// LLVM: store { i32, i32 } %[[TMP_A]], ptr %[[COMPLEX_B]], align 4

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: store i32 %[[A_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store i32 %[[A_IMAG]], ptr %[[B_IMAG_PTR]], align 4

int foo16(int _Complex a, int _Complex b) {
  return __imag__ a + __imag__ b;
}

// CIR: %[[RET:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[COMPLEX_A]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[B_IMAG:.*]] = cir.complex.imag %[[COMPLEX_B]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[ADD:.*]] = cir.binop(add, %[[A_IMAG]], %[[B_IMAG]]) nsw : !s32i
// CIR: cir.store %[[ADD]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[TMP]] : !s32i

// LLVM: %[[RET:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[COMPLEX_A:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[COMPLEX_A]], 1
// LLVM: %[[COMPLEX_B:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[COMPLEX_B]], 1
// LLVM: %[[ADD:.*]] = add nsw i32 %[[A_IMAG]], %[[B_IMAG]]
// LLVM: store i32 %[[ADD]], ptr %[[RET]], align 4
// LLVM: %[[TMP:.*]] = load i32, ptr %[[RET]], align 4
// LLVM: ret i32 %[[TMP]]

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_IMAG:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_IMAG]], align 4
// OGCG: %[[B_IMAG:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: %[[TMP_B:.*]] = load i32, ptr %[[B_IMAG]], align 4
// OGCG: %[[ADD:.*]] = add nsw i32 %[[TMP_A]], %[[TMP_B]]
// OGCG: ret i32 %[[ADD]]

int foo17(int _Complex a, int _Complex b) {
  return __real__ a + __real__ b;
}

// CIR: %[[RET:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[COMPLEX_A]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[B_REAL:.*]] = cir.complex.real %[[COMPLEX_B]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[ADD:.*]] = cir.binop(add, %[[A_REAL]], %[[B_REAL]]) nsw : !s32i
// CIR: cir.store %[[ADD]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[TMP]] : !s32i

// LLVM: %[[RET:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[COMPLEX_A:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[COMPLEX_A]], 0
// LLVM: %[[COMPLEX_B:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[COMPLEX_B]], 0
// LLVM: %[[ADD:.*]] = add nsw i32 %[[A_REAL]], %[[B_REAL]]
// LLVM: store i32 %[[ADD]], ptr %[[RET]], align 4
// LLVM: %[[TMP:.*]] = load i32, ptr %[[RET]], align 4
// LLVM: ret i32 %[[TMP]]

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_REAL]], align 4
// OGCG: %[[B_REAL:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[TMP_B:.*]] = load i32, ptr %[[B_REAL]], align 4
// OGCG: %[[ADD:.*]] = add nsw i32 %[[TMP_A]], %[[TMP_B]]
// OGCG: ret i32 %[[ADD]]

bool foo18(int _Complex a, int _Complex b) {
  return a == b;
}

// CIR: %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[RESULT:.*]] = cir.cmp(eq, %[[COMPLEX_A]], %[[COMPLEX_B]]) : !cir.complex<!s32i>, !cir.bool

// LLVM: %[[COMPLEX_A:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM: %[[COMPLEX_B:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[COMPLEX_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[COMPLEX_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[COMPLEX_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[COMPLEX_B]], 1
// LLVM: %[[CMP_REAL:.*]] = icmp eq i32 %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[CMP_IMAG:.*]] = icmp eq i32 %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[RESULT:.*]] = and i1 %[[CMP_REAL]], %[[CMP_IMAG]]

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[CMP_REAL:.*]] = icmp eq i32 %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[CMP_IMAG:.*]] = icmp eq i32 %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[RESULT:.*]] = and i1 %[[CMP_REAL]], %[[CMP_IMAG]]

bool foo19(double _Complex a, double _Complex b) {
  return a == b;
}

// CIR: %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: %[[RESULT:.*]] = cir.cmp(eq, %[[COMPLEX_A]], %[[COMPLEX_B]]) : !cir.complex<!cir.double>, !cir.bool


// LLVM: %[[COMPLEX_A:.*]] = load { double, double }, ptr {{.*}}, align 8
// LLVM: %[[COMPLEX_B:.*]] = load { double, double }, ptr {{.*}}, align 8
// LLVM: %[[A_REAL:.*]] = extractvalue { double, double } %[[COMPLEX_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { double, double } %[[COMPLEX_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { double, double } %[[COMPLEX_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { double, double } %[[COMPLEX_B]], 1
// LLVM: %[[CMP_REAL:.*]] = fcmp oeq double %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[CMP_IMAG:.*]] = fcmp oeq double %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[RESULT:.*]] = and i1 %[[CMP_REAL]], %[[CMP_IMAG]]

// OGCG: %[[COMPLEX_A:.*]] = alloca { double, double }, align 8
// OGCG: %[[COMPLEX_B:.*]] = alloca { double, double }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: store double {{.*}}, ptr %[[A_REAL_PTR]], align 8
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: store double {{.*}}, ptr %[[A_IMAG_PTR]], align 8
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: store double {{.*}}, ptr %[[B_REAL_PTR]], align 8
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: store double {{.*}}, ptr %[[B_IMAG_PTR]], align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load double, ptr %[[A_REAL_PTR]], align 8
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load double, ptr %[[A_IMAG_PTR]], align 8
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load double, ptr %[[B_REAL_PTR]], align 8
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load double, ptr %[[B_IMAG_PTR]], align 8
// OGCG: %[[CMP_REAL:.*]] = fcmp oeq double %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[CMP_IMAG:.*]] = fcmp oeq double %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[RESULT:.*]] = and i1 %[[CMP_REAL]], %[[CMP_IMAG]]


bool foo20(int _Complex a, int _Complex b) {
  return a != b;
}

// CIR: %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[RESULT:.*]] = cir.cmp(ne, %[[COMPLEX_A]], %[[COMPLEX_B]]) : !cir.complex<!s32i>, !cir.bool

// LLVM: %[[COMPLEX_A:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM: %[[COMPLEX_B:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[COMPLEX_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[COMPLEX_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[COMPLEX_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[COMPLEX_B]], 1
// LLVM: %[[CMP_REAL:.*]] = icmp ne i32 %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[CMP_IMAG:.*]] = icmp ne i32 %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[RESULT:.*]] = or i1 %[[CMP_REAL]], %[[CMP_IMAG]]

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[CMP_REAL:.*]] = icmp ne i32 %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[CMP_IMAG:.*]] = icmp ne i32 %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[RESULT:.*]] = or i1 %[[CMP_REAL]], %[[CMP_IMAG]]

bool foo21(double _Complex a, double _Complex b) {
  return a != b;
}

// CIR: %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: %[[RESULT:.*]] = cir.cmp(ne, %[[COMPLEX_A]], %[[COMPLEX_B]]) : !cir.complex<!cir.double>, !cir.bool

// LLVM: %[[COMPLEX_A:.*]] = load { double, double }, ptr {{.*}}, align 8
// LLVM: %[[COMPLEX_B:.*]] = load { double, double }, ptr {{.*}}, align 8
// LLVM: %[[A_REAL:.*]] = extractvalue { double, double } %[[COMPLEX_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { double, double } %[[COMPLEX_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { double, double } %[[COMPLEX_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { double, double } %[[COMPLEX_B]], 1
// LLVM: %[[CMP_REAL:.*]] = fcmp une double %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[CMP_IMAG:.*]] = fcmp une double %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[RESULT:.*]] = or i1 %[[CMP_REAL]], %[[CMP_IMAG]]

// OGCG: %[[COMPLEX_A:.*]] = alloca { double, double }, align 8
// OGCG: %[[COMPLEX_B:.*]] = alloca { double, double }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: store double {{.*}}, ptr %[[A_REAL_PTR]], align 8
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: store double {{.*}}, ptr %[[A_IMAG_PTR]], align 8
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: store double {{.*}}, ptr %[[B_REAL_PTR]], align 8
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: store double {{.*}}, ptr %[[B_IMAG_PTR]], align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load double, ptr %[[A_REAL_PTR]], align 8
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load double, ptr %[[A_IMAG_PTR]], align 8
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load double, ptr %[[B_REAL_PTR]], align 8
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load double, ptr %[[B_IMAG_PTR]], align 8
// OGCG: %[[CMP_REAL:.*]] = fcmp une double %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[CMP_IMAG:.*]] = fcmp une double %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[RESULT:.*]] = or i1 %[[CMP_REAL]], %[[CMP_IMAG]]

void foo22(int _Complex a, int _Complex b) {
  int _Complex c = (a, b);
}

// CIR: %[[COMPLEX_A:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX_B:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[COMPLEX_B]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP_B]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[COMPLEX_B]], align 4
// LLVM: store { i32, i32 } %[[TMP_B]], ptr %[[RESULT]], align 4

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[RESULT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store i32 %[[B_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store i32 %[[B_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo23(int _Complex a, int _Complex b) {
  float _Complex f;
  int _Complex c = _Generic(a, int _Complex: b, default: f);
}

// CIR: %[[COMPLEX_A:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX_B:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[COMPLEX_F:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["f"]
// CIR: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX_B]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_F:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { i32, i32 }, ptr %[[COMPLEX_B]], align 4
// LLVM: store { i32, i32 } %[[TMP]], ptr %[[RESULT]], align 4

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_F:.*]] = alloca { float, float }, align 4
// OGCG: %[[RESULT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store i32 %[[B_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store i32 %[[B_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo24() {
  int _Complex arr[2];
  int _Complex r = arr[1];
}

// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!cir.complex<!s32i> x 2>, !cir.ptr<!cir.array<!cir.complex<!s32i> x 2>>, ["arr"]
// CIR: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["r", init]
// CIR: %[[IDX:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!cir.complex<!s32i> x 2>> -> !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[RESULT_VAL:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!cir.complex<!s32i>>, %[[IDX]] : !s32i), !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[RESULT_VAL]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[ARR:.*]] = alloca [2 x { i32, i32 }], i64 1, align 16
// LLVM: %[[RESULT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[ARR_PTR:.*]] = getelementptr { i32, i32 }, ptr %[[ARR]], i32 0
// LLVM: %[[RESULT_VAL:.*]] = getelementptr { i32, i32 }, ptr %[[ARR_PTR]], i64 1
// LLVM: %[[TMP:.*]] = load { i32, i32 }, ptr %[[RESULT_VAL]], align 8
// LLVM: store { i32, i32 } %[[TMP]], ptr %[[RESULT]], align 4

// OGCG: %[[ARR:.*]] = alloca [2 x { i32, i32 }], align 16
// OGCG: %[[RESULT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[ELEM_PTR:.*]] = getelementptr inbounds [2 x { i32, i32 }], ptr %[[ARR]], i64 0, i64 1
// OGCG: %[[ELEM_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[ELEM_PTR]], i32 0, i32 0
// OGCG: %[[ELEM_REAL:.*]] = load i32, ptr %[[ELEM_REAL_PTR]]
// OGCG: %[[ELEM_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[ELEM_PTR]], i32 0, i32 1
// OGCG: %[[ELEM_IMAG:.*]] = load i32, ptr %[[ELEM_IMAG_PTR]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store i32 %[[ELEM_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store i32 %[[ELEM_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

template <double _Complex N> void template_foo() { double _Complex C = N; }

void foo25() {
  template_foo<__builtin_complex(1.0, 2.0)>();
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>, ["C", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.const_complex<#cir.fp<1.000000e+00> : !cir.double, #cir.fp<2.000000e+00> : !cir.double> : !cir.complex<!cir.double>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>

// LLVM: %[[INIT:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: store { double, double } { double 1.000000e+00, double 2.000000e+00 }, ptr %[[INIT]], align 8

// OGCG: %[[INIT:.*]] = alloca { double, double }, align 8
// OGCG: %[[INIT_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[INIT]], i32 0, i32 0
// OGCG: %[[INIT_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[INIT]], i32 0, i32 1
// OGCG: store double 1.000000e+00, ptr %[[INIT_REAL_PTR]], align 8
// OGCG: store double 2.000000e+00, ptr %[[INIT_IMAG_PTR]], align 8

void foo26(int _Complex* a) {
  int _Complex b = *a;
}

// CIR: %[[COMPLEX_A_PTR:.*]] = cir.alloca !cir.ptr<!cir.complex<!s32i>>, !cir.ptr<!cir.ptr<!cir.complex<!s32i>>>, ["a", init]
// CIR: %[[COMPLEX_B:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[COMPLEX_A:.*]] = cir.load deref {{.*}} %[[COMPLEX_A_PTR]] : !cir.ptr<!cir.ptr<!cir.complex<!s32i>>>, !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX_A]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP]], %[[COMPLEX_B]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COMPLEX_A_PTR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_A:.*]] = load ptr, ptr %[[COMPLEX_A_PTR]], align 8
// LLVM: %[[TMP:.*]] = load { i32, i32 }, ptr %[[COMPLEX_A]], align 4
// LLVM: store { i32, i32 } %[[TMP]], ptr %[[COMPLEX_B]], align 4

// OGCG: %[[COMPLEX_A_PTR:.*]] = alloca ptr, align 8
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_A:.*]] = load ptr, ptr %[[COMPLEX_A_PTR]], align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG: store i32 %[[A_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store i32 %[[A_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void foo27(bool cond, int _Complex a, int _Complex b) {
  int _Complex c = cond ? a : b;
}

// CIR: %[[COND:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cond", init]
// CIR: %[[COMPLEX_A:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX_B:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[TMP_COND:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[RESULT_VAL:.*]] = cir.ternary(%[[TMP_COND]], true {
// CIR:   %[[TMP_A:.*]] = cir.load{{.*}} %[[COMPLEX_A]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:   cir.yield %[[TMP_A]] : !cir.complex<!s32i>
// CIR: }, false {
// CIR:   %[[TMP_B:.*]] = cir.load{{.*}} %[[COMPLEX_B]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:   cir.yield %[[TMP_B]] : !cir.complex<!s32i>
// CIR: }) : (!cir.bool) -> !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RESULT_VAL]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COND:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_COND:.*]] = load i8, ptr %[[COND]], align 1
// LLVM: %[[COND_VAL:.*]] = trunc i8 %[[TMP_COND]] to i1
// LLVM: br i1 %[[COND_VAL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:  %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[COMPLEX_A]], align 4
// LLVM:  br label %[[END_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:  %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[COMPLEX_B]], align 4
// LLVM:  br label %[[END_BB]]
// LLVM: [[END_BB]]:
// LLVM: %[[RESULT_VAL:.*]] = phi { i32, i32 } [ %[[TMP_B]], %[[FALSE_BB]] ], [ %[[TMP_A]], %[[TRUE_BB]] ]
// LLVM: store { i32, i32 } %[[RESULT_VAL]], ptr %[[RESULT]], align 4

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COND:.*]] = alloca i8, align 1
// OGCG: %[[RESULT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[TMP_COND:.*]] = load i8, ptr %[[COND]], align 1
// OGCG: %[[COND_VAL:.*]] = trunc i8 %[[TMP_COND]] to i1
// OGCG: br i1 %[[COND_VAL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:  %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG:  %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG:  %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG:  %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG:  br label %[[END_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:  %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 0
// OGCG:  %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG:  %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_B]], i32 0, i32 1
// OGCG:  %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG:  br label %[[END_BB]]
// OGCG: [[END_BB]]:
// OGCG:  %[[REAL:.*]] = phi i32 [ %[[A_REAL]], %[[TRUE_BB]] ], [ %[[B_REAL]], %[[FALSE_BB]] ]
// OGCG:  %[[IMAG:.*]] = phi i32 [ %[[A_IMAG]], %[[TRUE_BB]] ], [ %[[B_IMAG]], %[[FALSE_BB]] ]
// OGCG:  %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 0
// OGCG:  %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 1
// OGCG:  store i32 %[[REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG:  store i32 %[[IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo28() {
  using IntComplex = int _Complex;
  int _Complex a = IntComplex();
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR: cir.store align(4) %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[INIT]], align 4

// OGCG: %[[INIT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[INIT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[INIT]], i32 0, i32 0
// OGCG: %[[INIT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[INIT]], i32 0, i32 1
// OGCG: store i32 0, ptr %[[INIT_REAL_PTR]], align 4
// OGCG: store i32 0, ptr %[[INIT_IMAG_PTR]], align 4

void foo29() {
  using IntComplex = int _Complex;
  int _Complex a = IntComplex{};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[INIT]], align 4

// OGCG: %[[INIT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[INIT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[INIT]], i32 0, i32 0
// OGCG: %[[INIT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[INIT]], i32 0, i32 1
// OGCG: store i32 0, ptr %[[INIT_REAL_PTR]], align 4
// OGCG: store i32 0, ptr %[[INIT_IMAG_PTR]], align 4

void foo30() {
  float _Complex a = { 1.0f };
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a", init]
// CIR: %[[CONST_1F:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CIR: %[[CONST_0F:.*]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[CONST_1F]], %[[CONST_0F]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: store { float, float } { float 1.000000e+00, float 0.000000e+00 }, ptr %[[A_ADDR]], align 4

// OGCG:  %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float 1.000000e+00, ptr %[[A_REAL_PTR]], align 4
// OGCG: store float 0.000000e+00, ptr %[[A_IMAG_PTR]], align 4

void foo31() {
  struct Wrapper {
    int _Complex c;
  };

  Wrapper w;
  int r = __real__ w.c;
}

// CIR: %[[W_ADDR:.*]] = cir.alloca !rec_Wrapper, !cir.ptr<!rec_Wrapper>, ["w"]
// CIR: %[[REAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init]
// CIR: %[[ELEM_PTR:.*]] = cir.get_member %[[W_ADDR]][0] {name = "c"} : !cir.ptr<!rec_Wrapper> -> !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP_ELEM_PTR:.*]] = cir.load{{.*}} %[[ELEM_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[REAL:.*]] = cir.complex.real %[[TMP_ELEM_PTR]] : !cir.complex<!s32i> -> !s32i
// CIR: cir.store{{.*}} %[[REAL]], %[[REAL_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[W_ADDR:.*]] = alloca %struct.Wrapper, i64 1, align 4
// LLVM: %[[REAL_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[ELEM_PTR:.*]] = getelementptr %struct.Wrapper, ptr %[[W_ADDR]], i32 0, i32 0
// LLVM: %[[TMP_ELEM_PTR:.*]] = load { i32, i32 }, ptr %[[ELEM_PTR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %[[TMP_ELEM_PTR]], 0
// LLVM: store i32 %[[REAL]], ptr %[[REAL_ADDR]], align 4

// OGCG: %[[W_ADDR:.*]] = alloca %struct.Wrapper, align 4
// OGCG: %[[REAL_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[ELEM_PTR:.*]] = getelementptr inbounds nuw %struct.Wrapper, ptr %[[W_ADDR]], i32 0, i32 0
// OGCG: %[[REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[ELEM_PTR]], i32 0, i32 0
// OGCG: %[[REAL:.*]] = load i32, ptr %[[REAL_PTR]], align 4
// OGCG: store i32 %[[REAL]], ptr %[[REAL_ADDR]], align 4

struct Container {
  static int _Complex c;
};

void foo32() {
  Container con;
  int r = __real__ con.c;
}

// CIR: %[[REAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init]
// CIR: %[[ELEM_PTR:.*]] = cir.get_global @_ZN9Container1cE : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[ELEM:.*]] = cir.load{{.*}} %[[ELEM_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[REAL:.*]] = cir.complex.real %[[ELEM]] : !cir.complex<!s32i> -> !s32i
// CIR: cir.store{{.*}} %[[REAL]], %[[REAL_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[REAL_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[ELEM:.*]] = load { i32, i32 }, ptr @_ZN9Container1cE, align 4
// LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %[[ELEM]], 0
// LLVM: store i32 %[[REAL]], ptr %[[REAL_ADDR]], align 4

// OGCG: %[[REAL_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[REAL:.*]] = load i32, ptr @_ZN9Container1cE, align 4
// OGCG: store i32 %[[REAL]], ptr %[[REAL_ADDR]], align 4

void foo33(__builtin_va_list a) {
  float _Complex b = __builtin_va_arg(a, float _Complex);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!rec___va_list_tag>, !cir.ptr<!cir.ptr<!rec___va_list_tag>>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: cir.store %[[ARG_0:.*]], %[[A_ADDR]] : !cir.ptr<!rec___va_list_tag>, !cir.ptr<!cir.ptr<!rec___va_list_tag>>
// CIR: %[[VA_TAG:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.ptr<!rec___va_list_tag>>, !cir.ptr<!rec___va_list_tag>
// CIR: %[[COMPLEX:.*]] = cir.va_arg %[[VA_TAG]] : (!cir.ptr<!rec___va_list_tag>) -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: store ptr %[[ARG_0:.*]], ptr %[[A_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// LLVM: %[[COMPLEX:.*]] = va_arg ptr %[[TMP_A]], { float, float }
// LLVM: store { float, float } %[[COMPLEX]], ptr %[[B_ADDR]], align 4

// TODO(CIR): the difference between the CIR LLVM and OGCG is because the lack of calling convention lowering,
// Test will be updated when that is implemented

// OGCG: %[[A_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: store ptr %[[ARG_0:.*]], ptr %[[A_ADDR]], align 8
// OGCG: %[[TMP_A:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// OGCG: %[[GP_OFFSET_PTR:.*]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[TMP_A]], i32 0, i32 1
// OGCG: %[[GP_OFFSET:.*]] = load i32, ptr %[[GP_OFFSET_PTR]], align 4
// OGCG: %[[COND:.*]] = icmp ule i32 %[[GP_OFFSET]], 160
// OGCG: br i1 %[[COND]], label %[[VA_ARG_IN_REG:.*]], label %[[VA_ARG_IN_MEM:.*]]
//
// OGCG: [[VA_ARG_IN_REG]]:
// OGCG:  %[[REG_SAVE_PTR:.*]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[TMP_A]], i32 0, i32 3
// OGCG:  %[[REG_SAVE:.*]] = load ptr, ptr %[[REG_SAVE_PTR]], align 8
// OGCG:  %[[VA_ADDR:..*]] = getelementptr i8, ptr %[[REG_SAVE]], i32 %[[GP_OFFSET]]
// OGCG:  %[[GP_OFFSET_NEXT:.*]] = add i32 %[[GP_OFFSET]], 16
// OGCG:  store i32 %[[GP_OFFSET_NEXT]], ptr %[[GP_OFFSET_PTR]], align 4
// OGCG:  br label %[[VA_ARG_END:.*]]
//
// OGCG: [[VA_ARG_IN_MEM]]:
// OGCG:  %[[OVERFLOW_PTR:.*]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[TMP_A]], i32 0, i32 2
// OGCG:  %[[OVERFLOW:.*]] = load ptr, ptr %[[OVERFLOW_PTR]], align 8
// OGCG:  %[[OVERFLOW_NEXT:.*]] = getelementptr i8, ptr %[[OVERFLOW]], i32 8
// OGCG:  store ptr %[[OVERFLOW_NEXT]], ptr %[[OVERFLOW_PTR]], align 8
// OGCG:  br label %[[VA_ARG_END]]
//
// OGCG: [[VA_ARG_END]]:
// OGCG:  %[[RESULT:.*]] = phi ptr [ %[[VA_ADDR]], %[[VA_ARG_IN_REG]] ], [ %[[OVERFLOW]], %[[VA_ARG_IN_MEM]] ]
// OGCG:  %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 0
// OGCG:  %[[RESULT_REAL:.*]] = load float, ptr %[[RESULT_REAL_PTR]], align 4
// OGCG:  %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 1
// OGCG:  %[[RESULT_IMAG:.*]] = load float, ptr %[[RESULT_IMAG_PTR]], align 4
// OGCG:  %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG:  %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG:  store float %[[RESULT_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG:  store float %[[RESULT_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void foo34() {
  _Atomic(float _Complex) a;
  __c11_atomic_init(&a, {1.0f, 2.0f});
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[CONST_COMPLEX:.*]] = cir.const #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[CONST_COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: store { float, float } { float 1.000000e+00, float 2.000000e+00 }, ptr %[[A_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float 1.000000e+00, ptr %[[A_REAL_PTR]], align 8
// OGCG: store float 2.000000e+00, ptr %[[A_IMAG_PTR]], align 4

void foo35() {
  _Float16 _Complex a;
  _Float16 real = __real__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR: %[[REAL_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["real", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_REAL_F32:.*]] = cir.cast floating %[[A_REAL]] : !cir.f16 -> !cir.float
// CIR: %[[A_IMAG_F32:.*]] = cir.cast floating %[[A_IMAG]] : !cir.f16 -> !cir.float
// CIR: %[[A_COMPLEX_F32:.*]] = cir.complex.create %[[A_REAL_F32]], %[[A_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[A_REAL_F32:.*]] = cir.complex.real %[[A_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_REAL_F16:.*]] = cir.cast floating %[[A_REAL_F32]] : !cir.float -> !cir.f16
// CIR: cir.store{{.*}} %[[A_REAL_F16]], %[[REAL_ADDR]] : !cir.f16, !cir.ptr<!cir.f16>

// LLVM: %[[A_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[REAL_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM: %[[TMP_A:.*]] = load { half, half }, ptr %[[A_ADDR]], align 2
// LLVM: %[[A_REAL:.*]] = extractvalue { half, half } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { half, half } %[[TMP_A]], 1
// LLVM: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// LLVM: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_A_COMPLEX_F32]], float %[[A_IMAG_F32]], 1
// LLVM: %[[A_REAL_F16:.*]] = fptrunc float %[[A_REAL_F32]] to half
// LLVM: store half %[[A_REAL_F16]], ptr %[[REAL_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[REAL_ADDR:.*]] = alloca half, align 2
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load half, ptr %[[A_REAL_PTR]], align 2
// OGCG: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// OGCG: %[[A_REAL_F16:.*]] = fptrunc float %[[A_REAL_F32]] to half
// OGCG: store half %[[A_REAL_F16]], ptr %[[REAL_ADDR]], align 2

void foo36() {
  _Float16 _Complex a;
  _Float16 imag = __imag__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR: %[[IMAG_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["imag", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_REAL_F32:.*]] = cir.cast floating %[[A_REAL]] : !cir.f16 -> !cir.float
// CIR: %[[A_IMAG_F32:.*]] = cir.cast floating %[[A_IMAG]] : !cir.f16 -> !cir.float
// CIR: %[[A_COMPLEX_F32:.*]] = cir.complex.create %[[A_REAL_F32]], %[[A_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[A_IMAG_F32:.*]] = cir.complex.imag %[[A_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG_F16:.*]] = cir.cast floating %[[A_IMAG_F32]] : !cir.float -> !cir.f16
// CIR: cir.store{{.*}} %[[A_IMAG_F16]], %[[IMAG_ADDR]] : !cir.f16, !cir.ptr<!cir.f16>

// LLVM: %[[A_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[IMAG_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM: %[[TMP_A:.*]] = load { half, half }, ptr %[[A_ADDR]], align 2
// LLVM: %[[A_REAL:.*]] = extractvalue { half, half } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { half, half } %[[TMP_A]], 1
// LLVM: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// LLVM: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_A_COMPLEX_F32]], float %[[A_IMAG_F32]], 1
// LLVM: %[[A_IMAG_F16:.*]] = fptrunc float %[[A_IMAG_F32]] to half
// LLVM: store half %[[A_IMAG_F16]], ptr %[[IMAG_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[IMAG_ADDR:.*]] = alloca half, align 2
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load half, ptr %[[A_IMAG_PTR]], align 2
// OGCG: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// OGCG: %[[A_IMAG_F16:.*]] = fptrunc float %[[A_IMAG_F32]] to half
// OGCG: store half %[[A_IMAG_F16]], ptr %[[IMAG_ADDR]], align 2

void foo37() {
  _Complex float a;
  _Complex float b = __extension__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[TMP_A]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: store { float, float } %[[TMP_A]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void real_on_non_glvalue() {
  float _Complex a;
  float b = __real__(+a);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_REAL_PLUS:.*]] = cir.unary(plus, %[[A_REAL]]) : !cir.float, !cir.float
// CIR: %[[A_IMAG_PLUS:.*]] = cir.unary(plus, %[[A_IMAG]]) : !cir.float, !cir.float
// CIR: %[[RESULT:.*]] = cir.complex.create %[[A_REAL_PLUS]], %[[A_IMAG_PLUS]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[RESULT_REAL:.*]] = cir.complex.real %[[RESULT]] : !cir.complex<!cir.float> -> !cir.float
// CIR: cir.store{{.*}} %[[RESULT_REAL]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[A_IMAG]], 1
// LLVM: store float %[[A_REAL]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: store float %[[A_REAL]], ptr %[[B_ADDR]], align 4

void imag_on_non_glvalue() {
  float _Complex a;
  float b = __imag__(+a);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_REAL_PLUS:.*]] = cir.unary(plus, %[[A_REAL]]) : !cir.float, !cir.float
// CIR: %[[A_IMAG_PLUS:.*]] = cir.unary(plus, %[[A_IMAG]]) : !cir.float, !cir.float
// CIR: %[[RESULT:.*]] = cir.complex.create %[[A_REAL_PLUS]], %[[A_IMAG_PLUS]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[RESULT_IMAG:.*]] = cir.complex.imag %[[RESULT]] : !cir.complex<!cir.float> -> !cir.float
// CIR: cir.store{{.*}} %[[RESULT_IMAG]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[A_IMAG]], 1
// LLVM: store float %[[A_IMAG]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[B_ADDR]], align 4

void atomic_complex_type() {
  _Atomic(float _Complex) a;
  float _Complex b = __c11_atomic_load(&a, __ATOMIC_RELAXED);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[ATOMIC_TMP_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["atomic-temp"]
// CIR: %[[A_PTR:.*]] = cir.cast bitcast %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!u64i>
// CIR: %[[ATOMIC_TMP_PTR:.*]] = cir.cast bitcast %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!u64i>
// CIR: %[[TMP_A_ATOMIC:.*]] = cir.load{{.*}} atomic(relaxed) %[[A_PTR]] : !cir.ptr<!u64i>, !u64i
// CIR: cir.store{{.*}} %[[TMP_A_ATOMIC]], %[[ATOMIC_TMP_PTR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_ATOMIC_PTR:.*]] = cir.cast bitcast %[[ATOMIC_TMP_PTR]] : !cir.ptr<!u64i> -> !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[TMP_ATOMIC:.*]] = cir.load{{.*}} %[[TMP_ATOMIC_PTR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[TMP_ATOMIC]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[ATOMIC_TMP_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: %[[TMP_A_ATOMIC:.*]] = load atomic i64, ptr %[[A_ADDR]] monotonic, align 8
// LLVM: store i64 %[[TMP_A_ATOMIC]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: %[[TMP_ATOMIC:.*]] = load { float, float }, ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: store { float, float } %[[TMP_ATOMIC]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[ATOMIC_TMP_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[TMP_A_ATOMIC:.*]] = load atomic i64, ptr %[[A_ADDR]] monotonic, align 8
// OGCG: store i64 %[[TMP_A_ATOMIC]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// OGCG: %[[ATOMIC_TMP_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 0
// OGCG: %[[ATOMIC_TMP_REAL:.*]] = load float, ptr %[[ATOMIC_TMP_REAL_PTR]], align 8
// OGCG: %[[ATOMIC_TMP_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 1
// OGCG: %[[ATOMIC_TMP_IMAG:.*]] = load float, ptr %[[ATOMIC_TMP_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[ATOMIC_TMP_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store float %[[ATOMIC_TMP_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void real_on_scalar_glvalue() {
  float a;
  float b = __real__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.float -> !cir.float
// CIR: cir.store{{.*}} %[[A_REAL]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load float, ptr %[[A_ADDR]], align 4
// LLVM: store float %[[TMP_A]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca float, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: %[[TMP_A:.*]] = load float, ptr %[[A_ADDR]], align 4
// OGCG: store float %[[TMP_A]], ptr %[[B_ADDR]], align 4

void imag_on_scalar_glvalue() {
  float a;
  float b = __imag__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load %[[A_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.float -> !cir.float
// CIR: cir.store{{.*}} %[[A_IMAG]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: store float 0.000000e+00, ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca float, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: store float 0.000000e+00, ptr %[[B_ADDR]], align 4

void real_on_scalar_with_type_promotion() {
  _Float16 a;
  _Float16 b = __real__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.f16>, !cir.f16
// CIR: %[[TMP_A_F32:.*]] = cir.cast floating %[[TMP_A]] : !cir.f16 -> !cir.float
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A_F32]] : !cir.float -> !cir.float
// CIR: %[[TMP_A_F16:.*]] = cir.cast floating %[[A_REAL]] : !cir.float -> !cir.f16
// CIR: cir.store{{.*}} %[[TMP_A_F16]], %[[B_ADDR]] : !cir.f16, !cir.ptr<!cir.f16>

// LLVM: %[[A_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM: %[[B_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM: %[[TMP_A:.*]] = load half, ptr %[[A_ADDR]], align 2
// LLVM: %[[TMP_A_F32:.*]] = fpext half %[[TMP_A]] to float
// LLVM: %[[TMP_A_F16:.*]] = fptrunc float %[[TMP_A_F32]] to half
// LLVM: store half %[[TMP_A_F16]], ptr %[[B_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca half, align 2
// OGCG: %[[B_ADDR:.*]] = alloca half, align 2
// OGCG: %[[TMP_A:.*]] = load half, ptr %[[A_ADDR]], align 2
// OGCG: %[[TMP_A_F32:.*]] = fpext half %[[TMP_A]] to float
// OGCG: %[[TMP_A_F16:.*]] = fptrunc float %[[TMP_A_F32]] to half
// OGCG: store half %[[TMP_A_F16]], ptr %[[B_ADDR]], align 2

void imag_on_scalar_with_type_promotion() {
  _Float16 a;
  _Float16 b = __imag__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load %[[A_ADDR]] : !cir.ptr<!cir.f16>, !cir.f16
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.f16 -> !cir.f16
// CIR: %[[A_IMAG_F16:.*]] = cir.cast floating %[[A_IMAG]] : !cir.f16 -> !cir.f16
// CIR: cir.store{{.*}} %[[A_IMAG_F16]], %[[B_ADDR]] : !cir.f16, !cir.ptr<!cir.f16>

// LLVM: %[[A_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM: %[[B_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM: store half 0xH0000, ptr %[[B_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca half, align 2
// OGCG: %[[B_ADDR:.*]] = alloca half, align 2
// OGCG: store half 0xH0000, ptr %[[B_ADDR]], align 2

void imag_on_const_scalar() {
  float a;
  float b = __imag__ 1.0f;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[CONST_ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CIR: %[[CONST_IMAG:.*]] = cir.complex.imag %[[CONST_ONE]] : !cir.float -> !cir.float
// CIR: cir.store{{.*}} %[[CONST_IMAG]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: store float 0.000000e+00, ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca float, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: store float 0.000000e+00, ptr %[[B_ADDR]], align 4

void real_on_scalar_from_real_with_type_promotion() {
  _Float16 _Complex a;
  _Float16 b = __real__(__real__ a);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_REAL_F32:.*]] = cir.cast floating %[[A_REAL]] : !cir.f16 -> !cir.float
// CIR: %[[A_IMAG_F32:.*]] = cir.cast floating %[[A_IMAG]] : !cir.f16 -> !cir.float
// CIR: %[[A_COMPLEX_F32:.*]] = cir.complex.create %[[A_REAL_F32]], %[[A_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[A_REAL_F32:.*]] = cir.complex.real %[[A_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[A_REAL_F32]] : !cir.float -> !cir.float
// CIR: %[[A_REAL_F16:.*]] = cir.cast floating %[[A_REAL]] : !cir.float -> !cir.f16
// CIR: cir.store{{.*}} %[[A_REAL_F16]], %[[B_ADDR]] : !cir.f16, !cir.ptr<!cir.f16>

// LLVM: %[[A_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[B_ADDR]] = alloca half, i64 1, align 2
// LLVM: %[[TMP_A:.*]] = load { half, half }, ptr %[[A_ADDR]], align 2
// LLVM: %[[A_REAL:.*]] = extractvalue { half, half } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { half, half } %[[TMP_A]], 1
// LLVM: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// LLVM: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_A_COMPLEX_F32]], float %[[A_IMAG_F32]], 1
// LLVM: %[[A_REAL_F16:.*]] = fptrunc float %[[A_REAL_F32]] to half
// LLVM: store half %[[A_REAL_F16]], ptr %[[B_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[B_ADDR:.*]] = alloca half, align 2
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load half, ptr %[[A_REAL_PTR]], align 2
// OGCG: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// OGCG: %[[A_REAL_F16:.*]] = fptrunc float %[[A_REAL_F32]] to half
// OGCG: store half %[[A_REAL_F16]], ptr %[[B_ADDR]], align 2

void real_on_scalar_from_imag_with_type_promotion() {
  _Float16 _Complex a;
  _Float16 b = __real__(__imag__ a);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_REAL_F32:.*]] = cir.cast floating %[[A_REAL]] : !cir.f16 -> !cir.float
// CIR: %[[A_IMAG_F32:.*]] = cir.cast floating %[[A_IMAG]] : !cir.f16 -> !cir.float
// CIR: %[[A_COMPLEX_F32:.*]] = cir.complex.create %[[A_REAL_F32]], %[[A_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[A_IMAG_F32:.*]] = cir.complex.imag %[[A_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_REAL_F32:.*]] = cir.complex.real %[[A_IMAG_F32]] : !cir.float -> !cir.float
// CIR: %[[A_REAL_F16:.*]] = cir.cast floating %[[A_REAL_F32]] : !cir.float -> !cir.f16
// CIR: cir.store{{.*}} %[[A_REAL_F16]], %[[B_ADDR]] : !cir.f16, !cir.ptr<!cir.f16>

// LLVM: %[[A_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[B_ADDR]] = alloca half, i64 1, align 2
// LLVM: %[[TMP_A:.*]] = load { half, half }, ptr %[[A_ADDR]], align 2
// LLVM: %[[A_REAL:.*]] = extractvalue { half, half } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { half, half } %[[TMP_A]], 1
// LLVM: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// LLVM: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_A_COMPLEX_F32]], float %[[A_IMAG_F32]], 1
// LLVM: %[[A_IMAG_F16:.*]] = fptrunc float %[[A_IMAG_F32]] to half
// LLVM: store half %[[A_IMAG_F16]], ptr %[[B_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[B_ADDR:.*]] = alloca half, align 2
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load half, ptr %[[A_IMAG_PTR]], align 2
// OGCG: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// OGCG: %[[A_IMAG_F16:.*]] = fptrunc float %[[A_IMAG_F32]] to half
// OGCG: store half %[[A_IMAG_F16]], ptr %[[B_ADDR]], align 2

void complex_type_parameter(float _Complex a) {}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a", init]
// CIR: cir.store %{{.*}}, %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// TODO(CIR): the difference between the CIR LLVM and OGCG is because the lack of calling convention lowering,
// Test will be updated when that is implemented

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: store { float, float } %{{.*}}, ptr %[[A_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: store <2 x float> %a.coerce, ptr %[[A_ADDR]], align 4

void complex_type_argument() {
  float _Complex a;
  complex_type_parameter(a);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[ARG_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["coerce"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[TMP_A]], %[[ARG_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[TMP_ARG:.*]] = cir.load{{.*}} %[[ARG_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: cir.call @_Z22complex_type_parameterCf(%[[TMP_ARG]]) : (!cir.complex<!cir.float>) -> ()

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[ARG_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: store { float, float } %[[TMP_A]], ptr %[[ARG_ADDR]], align 4
// LLVM: %[[TMP_ARG:.*]] = load { float, float }, ptr %[[ARG_ADDR]], align 4
// LLVM: call void @_Z22complex_type_parameterCf({ float, float } %[[TMP_ARG]])

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[ARG_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[ARG_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[ARG_ADDR]], i32 0, i32 0
// OGCG: %[[ARG_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[ARG_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL]], ptr %[[ARG_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[ARG_IMAG_PTR]], align 4
// OGCG: %[[TMP_ARG:.*]] = load <2 x float>, ptr %[[ARG_ADDR]], align 4
// OGCG: call void @_Z22complex_type_parameterCf(<2 x float> noundef %[[TMP_ARG]])
