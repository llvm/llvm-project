// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
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
// CIR: %[[COMPLEX:.*]] = cir.const #cir.const_complex<#cir.int<0> : !s32i, #cir.int<0> : !s32i> : !cir.complex<!s32i>
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
