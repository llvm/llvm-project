// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void complex_functional_cast() {
  using IntComplex = int _Complex;
  int _Complex a = IntComplex{};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[INIT]], align 4

void complex_deref_expr(int _Complex* a) {
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

void complex_cxx_scalar_value_init_expr() {
  using IntComplex = int _Complex;
  int _Complex a = IntComplex();
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR: cir.store align(4) %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[INIT]], align 4

void complex_abstract_condition(bool cond, int _Complex a, int _Complex b) {
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

int _Complex complex_real_operator_on_rvalue() {
  int real = __real__ complex_real_operator_on_rvalue();
  return {};
}

// CIR: %[[RET_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["__retval"]
// CIR: %[[REAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["real", init]
// CIR: %[[CALL:.*]] = cir.call @_Z31complex_real_operator_on_rvaluev() : () -> !cir.complex<!s32i>
// CIR: %[[REAL:.*]] = cir.complex.real %[[CALL]] : !cir.complex<!s32i> -> !s32i
// CIR: cir.store{{.*}} %[[REAL]], %[[REAL_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[RET_COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RET_COMPLEX]], %[[RET_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP_RET:.*]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.return %[[TMP_RET]] : !cir.complex<!s32i>

// LLVM: %[[RET_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[REAL_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[CALL:.*]] = call { i32, i32 } @_Z31complex_real_operator_on_rvaluev()
// LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %[[CALL]], 0
// LLVM: store i32 %[[REAL]], ptr %[[REAL_ADDR]], align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[RET_ADDR]], align 4
// LLVM: %[[TMP_RET:.*]] = load { i32, i32 }, ptr %[[RET_ADDR]], align 4
// LLVM: ret { i32, i32 } %[[TMP_RET]]

void complex_member_expr() {
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

int _Complex complex_imag_operator_on_rvalue() {
  int imag = __imag__ complex_imag_operator_on_rvalue();
  return {};
}

// CIR: %[[RET_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["__retval"]
// CIR: %[[IMAG_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["imag", init]
// CIR: %[[CALL:.*]] = cir.call @_Z31complex_imag_operator_on_rvaluev() : () -> !cir.complex<!s32i>
// CIR: %[[IMAG:.*]] = cir.complex.imag %[[CALL]] : !cir.complex<!s32i> -> !s32i
// CIR: cir.store{{.*}} %[[IMAG]], %[[IMAG_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[RET_COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RET_COMPLEX]], %[[RET_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP_RET:.*]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.return %[[TMP_RET]] : !cir.complex<!s32i>

// LLVM: %[[RET_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[IMAG_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[CALL:.*]] = call { i32, i32 } @_Z31complex_imag_operator_on_rvaluev()
// LLVM: %[[IMAG:.*]] = extractvalue { i32, i32 } %[[CALL]], 1
// LLVM: store i32 %[[IMAG]], ptr %[[IMAG_ADDR]], align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[RET_ADDR]], align 4
// LLVM: %[[TMP_RET:.*]] = load { i32, i32 }, ptr %[[RET_ADDR]], align 4
// LLVM: ret { i32, i32 } %[[TMP_RET]]

struct Container {
  static int _Complex c;
};

void complex_member_expr_with_var_deal() {
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

void complex_comma_operator(int _Complex a, int _Complex b) {
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

void complex_cxx_default_init_expr() {
  struct FPComplexWrapper {
    float _Complex c{};
  };

  FPComplexWrapper w{};
}

// CIR: %[[W_ADDR:.*]] = cir.alloca !rec_FPComplexWrapper, !cir.ptr<!rec_FPComplexWrapper>, ["w", init]
// CIR: %[[C_ADDR:.*]] = cir.get_member %[[W_ADDR]][0] {name = "c"} : !cir.ptr<!rec_FPComplexWrapper> -> !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[CONST_COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[CONST_COMPLEX]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[W_ADDR:.*]] = alloca %struct.FPComplexWrapper, i64 1, align 4
// LLVM: %[[C_ADDR:.*]] = getelementptr %struct.FPComplexWrapper, ptr %[[W_ADDR]], i32 0, i32 0
// LLVM: store { float, float } zeroinitializer, ptr %[[C_ADDR]], align 4

// OGCG: %[[W_ADDR:.*]] = alloca %struct.Wrapper, align 4
// OGCG: %[[C_ADDR:.*]] = getelementptr inbounds nuw %struct.FPComplexWrapper, ptr %[[W_ADDR]], i32 0, i32 0
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG: store float 0.000000e+00, ptr %[[C_REAL_PTR]], align 4
// OGCG: store float 0.000000e+00, ptr %[[C_IMAG_PTR]], align 4

void complex_init_atomic() {
  _Atomic(float _Complex) a;
  __c11_atomic_init(&a, {1.0f, 2.0f});
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[CONST_COMPLEX:.*]] = cir.const #cir.complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[CONST_COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: store { float, float } { float 1.000000e+00, float 2.000000e+00 }, ptr %[[A_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float 1.000000e+00, ptr %[[A_REAL_PTR]], align 8
// OGCG: store float 2.000000e+00, ptr %[[A_IMAG_PTR]], align 4

void complex_opaque_value_expr() {
  float _Complex a;
  float b = 1.0f ?: __real__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CIR: cir.store align(4) %[[CONST_1]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: store float 1.000000e+00, ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: store float 1.000000e+00, ptr %[[B_ADDR]], align 4

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

void function_with_complex_default_arg(
    float _Complex a = __builtin_complex(1.0f, 2.2f)) {}

// CIR: %[[ARG_0_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a", init]
// CIR: cir.store %{{.*}}, %[[ARG_0_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// TODO(CIR): the difference between the CIR LLVM and OGCG is because the lack of calling convention lowering,

// LLVM: %[[ARG_0_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: store { float, float } %{{.*}}, ptr %[[ARG_0_ADDR]], align 4

// OGCG: %[[ARG_0_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: store <2 x float> %{{.*}}, ptr %[[ARG_0_ADDR]], align 4

void calling_function_with_default_arg() {
  function_with_complex_default_arg();
}

// CIR: %[[DEFAULT_ARG_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["coerce"]
// CIR: %[[DEFAULT_ARG_VAL:.*]] = cir.const #cir.complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.200000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[DEFAULT_ARG_VAL]], %[[DEFAULT_ARG_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[TMP_DEFAULT_ARG:.*]] = cir.load{{.*}} %[[DEFAULT_ARG_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: cir.call @_Z33function_with_complex_default_argCf(%[[TMP_DEFAULT_ARG]]) : (!cir.complex<!cir.float>) -> ()

// TODO(CIR): the difference between the CIR LLVM and OGCG is because the lack of calling convention lowering,

// LLVM: %[[DEFAULT_ARG_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: store { float, float } { float 1.000000e+00, float 0x40019999A0000000 }, ptr %[[DEFAULT_ARG_ADDR]], align 4
// LLVM: %[[TMP_DEFAULT_ARG:.*]] = load { float, float }, ptr %[[DEFAULT_ARG_ADDR]], align 4
// LLVM: call void @_Z33function_with_complex_default_argCf({ float, float } %[[TMP_DEFAULT_ARG]])

// OGCG: %[[DEFAULT_ARG_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[DEFAULT_ARG_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[DEFAULT_ARG_ADDR]], i32 0, i32 0
// OGCG: %[[DEFAULT_ARG_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[DEFAULT_ARG_ADDR]], i32 0, i32 1
// OGCG: store float 1.000000e+00, ptr %[[DEFAULT_ARG_REAL_PTR]], align 4
// OGCG: store float 0x40019999A0000000, ptr %[[DEFAULT_ARG_IMAG_PTR]], align 4
// OGCG: %[[TMP_DEFAULT_ARG:.*]] = load <2 x float>, ptr %[[DEFAULT_ARG_ADDR]], align 4
// OGCG: call void @_Z33function_with_complex_default_argCf(<2 x float> {{.*}} %[[TMP_DEFAULT_ARG]])

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
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.float -> !cir.float
// CIR: cir.store{{.*}} %[[A_IMAG]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load float, ptr %[[A_ADDR]], align 4
// LLVM: store float 0.000000e+00, ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca float, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: store float 0.000000e+00, ptr %[[B_ADDR]], align 4

void real_on_scalar_bool() {
  bool a;
  bool b = __real__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.bool -> !cir.bool
// CIR: cir.store{{.*}} %[[A_REAL]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// LLVM: %[[TMP_A_I1:.*]] = trunc i8 %[[TMP_A]] to i1
// LLVM: %[[TMP_A_I8:.*]] = zext i1 %[[TMP_A_I1]] to i8
// LLVM: store i8 %[[TMP_A_I8]], ptr %[[B_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[B_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// OGCG: %[[TMP_A_I1:.*]] = trunc i8 %[[TMP_A]] to i1
// OGCG: %[[TMP_A_I8:.*]] = zext i1 %[[TMP_A_I1]] to i8
// OGCG: store i8 %[[TMP_A_I8]], ptr %[[B_ADDR]], align 1

void imag_on_scalar_bool() {
  bool a;
  bool b = __imag__ a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.bool -> !cir.bool
// CIR: cir.store{{.*}} %[[A_IMAG]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// LLVM: %[[TMP_A_I1:.*]] = trunc i8 %[[TMP_A]] to i1
// LLVM: store i8 0, ptr %[[B_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[B_ADDR:.*]] = alloca i8, align 1
// OGCG: store i8 0, ptr %[[B_ADDR]], align 1
