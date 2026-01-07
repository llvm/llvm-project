// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
  int a;
  int b = 1 ?: a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST_1]], %[[B_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 1, ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 1, ptr %[[B_ADDR]], align 4

void foo2() {
  float _Complex a;
  float _Complex b;
  float _Complex c = a ?: b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_REAL_BOOL:.*]] = cir.cast float_to_bool %[[A_REAL]] : !cir.float -> !cir.bool
// CIR: %[[A_IMAG_BOOL:.*]] = cir.cast float_to_bool %[[A_IMAG]] : !cir.float -> !cir.bool
// CIR: %[[CONST_TRUE:.*]] = cir.const #true
// CIR: %[[COND:.*]] = cir.select if %[[A_REAL_BOOL]] then %[[CONST_TRUE]] else %[[A_IMAG_BOOL]] : (!cir.bool, !cir.bool, !cir.bool) -> !cir.bool
// CIR: %[[RESULT:.*]] = cir.ternary(%[[COND]], true {
// CIR:   %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   cir.yield %[[TMP_A]] : !cir.complex<!cir.float>
// CIR: }, false {
// CIR:   %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   cir.yield %[[TMP_B]] : !cir.complex<!cir.float>
// CIR: }) : (!cir.bool) -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[A_REAL_BOOL:.*]] = fcmp une float %[[A_REAL]], 0.000000e+00
// LLVM: %[[A_IMAG_BOOL:.*]] = fcmp une float %[[A_IMAG]], 0.000000e+00
// LLVM: %[[COND:.*]] = or i1 %[[A_REAL_BOOL]], %[[A_IMAG_BOOL]]
// LLVM: br i1 %[[COND]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// LLVM: [[COND_TRUE]]:
// LLVM:  %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM:  br label %[[COND_RESULT:.*]]
// LLVM: [[COND_FALSE]]:
// LLVM:  %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM:  br label %[[COND_RESULT]]
// LLVM: [[COND_RESULT]]:
// LLVM:  %[[RESULT:.*]] = phi { float, float } [ %[[TMP_B]], %[[COND_FALSE]] ], [ %[[TMP_A]], %[[COND_TRUE]] ]
// LLVM:  br label %[[COND_END:.*]]
// LLVM: [[COND_END]]:
// LLVM:  store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_BOOL:.*]] = fcmp une float %[[A_REAL]], 0.000000e+00
// OGCG: %[[A_IMAG_BOOL:.*]] = fcmp une float %[[A_IMAG]], 0.000000e+00
// OGCG: %[[COND:.*]] = or i1 %[[A_REAL_BOOL]], %[[A_IMAG_BOOL]]
// OGCG: br i1 %tobool2, label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// OGCG: [[COND_TRUE]]:
// OGCG:  %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG:  %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG:  %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG:  %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG:  br label %[[COND_END:.*]]
// OGCG: [[COND_FALSE]]:
// OGCG:  %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG:  %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG:  %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG:  %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG:  br label %[[COND_END]]
// OGCG: [[COND_END]]:
// OGCG:  %[[RESULT_REAL:.*]] = phi float [ %[[A_REAL]], %[[COND_TRUE]] ], [ %[[B_REAL]], %[[COND_FALSE]] ]
// OGCG:  %[[RESULT_IMAG:.*]] = phi float [ %[[A_IMAG]], %[[COND_TRUE]] ], [ %[[B_IMAG]], %[[COND_FALSE]] ]
// OGCG:  %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG:  %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG:  store float %[[RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG:  store float %[[RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4

void foo3() {
  int a;
  int b;
  int c = a ?: b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b"]
// CIR: %[[C_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[A_BOOL:.*]] = cir.cast int_to_bool %[[TMP_A]] : !s32i -> !cir.bool
// CIR: %[[RESULT:.*]] = cir.ternary(%[[A_BOOL]], true {
// CIR:   %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.yield %[[TMP_A]] : !s32i
// CIR: }, false {
// CIR:   %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.yield %[[TMP_B]] : !s32i
// CIR: }) : (!cir.bool) -> !s32i
// CIR: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[C_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM: %[[COND:.*]] = icmp ne i32 %[[TMP_A]], 0
// LLVM: br i1 %[[COND]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// LLVM: [[COND_TRUE]]:
// LLVM:  %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:  br label %[[COND_RESULT:.*]]
// LLVM: [[COND_FALSE]]:
// LLVM:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:  br label %[[COND_RESULT]]
// LLVM: [[COND_RESULT]]:
// LLVM:  %[[RESULT:.*]] = phi i32 [ %[[TMP_B]], %[[COND_FALSE]] ], [ %[[TMP_A]], %[[COND_TRUE]] ]
// LLVM:  br label %[[COND_END:.*]]
// LLVM: [[COND_END]]:
// LLVM:  store i32 %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[C_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: %[[A_BOOL:.*]] = icmp ne i32 %[[TMP_A]], 0
// OGCG: br i1 %[[A_BOOL]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// OGCG: [[COND_TRUE]]:
// OGCG:  %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:  br label %[[COND_END:.*]]
// OGCG: [[COND_FALSE]]:
// OGCG:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG:  br label %[[COND_END]]
// OGCG: [[COND_END]]:
// OGCG:  %[[RESULT:.*]] = phi i32 [ %[[TMP_A]], %[[COND_TRUE]] ], [ %[[TMP_B]], %[[COND_FALSE]] ]
// OGCG:  store i32 %[[RESULT]], ptr %[[C_ADDR]], align 4

void test_gnu_binary_lvalue_assign() {
  int a = 5;
  int b = 10;
  (a ?: b) = 42;
}

// CIR-LABEL: cir.func{{.*}} @_Z29test_gnu_binary_lvalue_assignv(
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[A_VAL:.*]] = cir.load{{.*}} %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[A_BOOL:.*]] = cir.cast int_to_bool %[[A_VAL]] : !s32i -> !cir.bool
// CIR: %[[TERNARY_PTR:.*]] = cir.ternary(%[[A_BOOL]], true {
// CIR:   cir.yield %[[A]] : !cir.ptr<!s32i>
// CIR: }, false {
// CIR:   cir.yield %[[B]] : !cir.ptr<!s32i>
// CIR: }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %{{.*}}, %[[TERNARY_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} void @_Z29test_gnu_binary_lvalue_assignv(
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM: %[[A_VAL:.*]] = load i32, ptr %[[A]]
// LLVM: %[[COND:.*]] = icmp ne i32 %[[A_VAL]], 0
// LLVM: br i1 %[[COND]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   br label %[[MERGE_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   br label %[[MERGE_BB]]
// LLVM: [[MERGE_BB]]:
// LLVM:   %[[PHI_PTR:.*]] = phi ptr [ %[[B]], %[[FALSE_BB]] ], [ %[[A]], %[[TRUE_BB]] ]
// LLVM:   br label %[[CONT_BB:.*]]
// LLVM: [[CONT_BB]]:
// LLVM:   store i32 42, ptr %[[PHI_PTR]]

// OGCG-LABEL: define{{.*}} void @_Z29test_gnu_binary_lvalue_assignv(
// OGCG: %[[A:.*]] = alloca i32
// OGCG: %[[B:.*]] = alloca i32
// OGCG: %[[A_VAL:.*]] = load i32, ptr %[[A]]
// OGCG: %[[COND:.*]] = icmp ne i32 %[[A_VAL]], 0
// OGCG: br i1 %[[COND]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   br label %[[MERGE_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   br label %[[MERGE_BB]]
// OGCG: [[MERGE_BB]]:
// OGCG:   %[[PHI_PTR:.*]] = phi ptr [ %[[A]], %[[TRUE_BB]] ], [ %[[B]], %[[FALSE_BB]] ]
// OGCG:   store i32 42, ptr %[[PHI_PTR]]

void test_gnu_binary_lvalue_compound() {
  int a = 7;
  int b = 14;
  (a ?: b) += 5;
}

// CIR-LABEL: cir.func{{.*}} @_Z31test_gnu_binary_lvalue_compoundv(
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[A_VAL:.*]] = cir.load{{.*}} %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[A_BOOL:.*]] = cir.cast int_to_bool %[[A_VAL]] : !s32i -> !cir.bool
// CIR: %[[LVAL_PTR:.*]] = cir.ternary(%[[A_BOOL]], true {
// CIR:   cir.yield %[[A]] : !cir.ptr<!s32i>
// CIR: }, false {
// CIR:   cir.yield %[[B]] : !cir.ptr<!s32i>
// CIR: }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR: %[[OLD_VAL:.*]] = cir.load{{.*}} %[[LVAL_PTR]]
// CIR: %[[NEW_VAL:.*]] = cir.binop(add, %[[OLD_VAL]], %{{.*}})
// CIR: cir.store{{.*}} %[[NEW_VAL]], %[[LVAL_PTR]]

// LLVM-LABEL: define{{.*}} void @_Z31test_gnu_binary_lvalue_compoundv(
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM: %[[A_VAL:.*]] = load i32, ptr %[[A]]
// LLVM: %[[COND:.*]] = icmp ne i32 %[[A_VAL]], 0
// LLVM: br i1 %[[COND]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   br label %[[MERGE_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   br label %[[MERGE_BB]]
// LLVM: [[MERGE_BB]]:
// LLVM:   %[[PTR:.*]] = phi ptr [ %[[B]], %[[FALSE_BB]] ], [ %[[A]], %[[TRUE_BB]] ]
// LLVM:   br label %[[CONT:.*]]
// LLVM: [[CONT]]:
// LLVM:   %[[OLD:.*]] = load i32, ptr %[[PTR]]
// LLVM:   %[[NEW:.*]] = add{{.*}} i32 %[[OLD]], 5
// LLVM:   store i32 %[[NEW]], ptr %[[PTR]]

// OGCG-LABEL: define{{.*}} void @_Z31test_gnu_binary_lvalue_compoundv(
// OGCG: %[[A:.*]] = alloca i32
// OGCG: %[[B:.*]] = alloca i32
// OGCG: %[[A_VAL:.*]] = load i32, ptr %[[A]]
// OGCG: %[[COND:.*]] = icmp ne i32 %[[A_VAL]], 0
// OGCG: br i1 %[[COND]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   br label %[[MERGE_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   br label %[[MERGE_BB]]
// OGCG: [[MERGE_BB]]:
// OGCG:   %[[PTR:.*]] = phi ptr [ %[[A]], %[[TRUE_BB]] ], [ %[[B]], %[[FALSE_BB]] ]
// OGCG:   %[[OLD:.*]] = load i32, ptr %[[PTR]]
// OGCG:   %[[NEW:.*]] = add{{.*}} i32 %[[OLD]], 5
// OGCG:   store i32 %[[NEW]], ptr %[[PTR]]

void test_gnu_binary_lvalue_ptr() {
  int x = 1, y = 2;
  int *p = &x;
  int *q = nullptr;
  *(p ?: q) = 99;
}

// CIR-LABEL: cir.func{{.*}} @_Z26test_gnu_binary_lvalue_ptrv(
// CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR: %[[Y:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
// CIR: %[[P:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init]
// CIR: %[[Q:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["q", init]
// CIR: %[[P_VAL:.*]] = cir.load{{.*}} %[[P]]
// CIR: %[[P_BOOL:.*]] = cir.cast ptr_to_bool %[[P_VAL]]
// CIR: %[[PTR_RESULT:.*]] = cir.ternary(%[[P_BOOL]], true {
// CIR:   %[[P_LOAD:.*]] = cir.load{{.*}} %[[P]]
// CIR:   cir.yield %[[P_LOAD]] : !cir.ptr<!s32i>
// CIR: }, false {
// CIR:   %[[Q_LOAD:.*]] = cir.load{{.*}} %[[Q]]
// CIR:   cir.yield %[[Q_LOAD]] : !cir.ptr<!s32i>
// CIR: }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %{{.*}}, %[[PTR_RESULT]]

// LLVM-LABEL: define{{.*}} void @_Z26test_gnu_binary_lvalue_ptrv(
// LLVM: %[[X:.*]] = alloca i32
// LLVM: %[[Y:.*]] = alloca i32
// LLVM: %[[P:.*]] = alloca ptr
// LLVM: %[[Q:.*]] = alloca ptr
// LLVM: %[[P_VAL:.*]] = load ptr, ptr %[[P]]
// LLVM: %[[COND:.*]] = icmp ne ptr %[[P_VAL]], null
// LLVM: br i1 %[[COND]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[P_LOAD:.*]] = load ptr, ptr %[[P]]
// LLVM:   br label %[[MERGE_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   %[[Q_LOAD:.*]] = load ptr, ptr %[[Q]]
// LLVM:   br label %[[MERGE_BB]]
// LLVM: [[MERGE_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[Q_LOAD]], %[[FALSE_BB]] ], [ %[[P_LOAD]], %[[TRUE_BB]] ]
// LLVM:   br label %[[CONT:.*]]
// LLVM: [[CONT]]:
// LLVM:   store i32 99, ptr %[[PHI]]

// OGCG-LABEL: define{{.*}} void @_Z26test_gnu_binary_lvalue_ptrv(
// OGCG: %[[X:.*]] = alloca i32
// OGCG: %[[Y:.*]] = alloca i32
// OGCG: %[[P:.*]] = alloca ptr
// OGCG: %[[Q:.*]] = alloca ptr
// OGCG: %[[P_VAL:.*]] = load ptr, ptr %[[P]]
// OGCG: %[[COND:.*]] = icmp ne ptr %[[P_VAL]], null
// OGCG: br i1 %[[COND]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   %[[P_LOAD:.*]] = load ptr, ptr %[[P]]
// OGCG:   br label %[[MERGE_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   %[[Q_LOAD:.*]] = load ptr, ptr %[[Q]]
// OGCG:   br label %[[MERGE_BB]]
// OGCG: [[MERGE_BB]]:
// OGCG:   %[[PHI:.*]] = phi ptr [ %[[P_LOAD]], %[[TRUE_BB]] ], [ %[[Q_LOAD]], %[[FALSE_BB]] ]
// OGCG:   store i32 99, ptr %[[PHI]]
