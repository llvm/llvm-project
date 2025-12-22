// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

int x(int y) {
  return y > 0 ? 3 : 5;
}

// CIR-LABEL: cir.func{{.*}} @_Z1xi(
// CIR-SAME: %[[ARG0:.*]]: !s32i {{.*}}) -> !s32i{{.*}}{
// CIR: [[Y:%.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
// CIR: [[RETVAL:%.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: cir.store %[[ARG0]], [[Y]] : !s32i, !cir.ptr<!s32i>
// CIR: [[YVAL:%.+]] = cir.load{{.*}} [[Y]] : !cir.ptr<!s32i>, !s32i
// CIR: [[ZERO:%.+]] = cir.const #cir.int<0> : !s32i
// CIR: [[CMP:%.+]] = cir.cmp(gt, [[YVAL]], [[ZERO]]) : !s32i, !cir.bool
// CIR: [[THREE:%.+]] = cir.const #cir.int<3> : !s32i
// CIR: [[FIVE:%.+]] = cir.const #cir.int<5> : !s32i
// CIR: [[SELECT_RES:%.+]] = cir.select if [[CMP]] then [[THREE]] else [[FIVE]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR: cir.store [[SELECT_RES]], [[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR: [[RETVAL_VAL:%.+]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return [[RETVAL_VAL]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z1xi(
// LLVM-SAME: i32 %[[ARG0:.+]])
// LLVM: %[[Y:.*]] = alloca i32
// LLVM: %[[RETVAL:.*]] = alloca i32
// LLVM: store i32 %[[ARG0]], ptr %[[Y]]
// LLVM: %[[YVAL:.*]] = load i32, ptr %[[Y]]
// LLVM: %[[CMP:.*]] = icmp sgt i32 %[[YVAL]], 0
// LLVM: %[[SELECT:.*]] = select i1 %[[CMP]], i32 3, i32 5
// LLVM: store i32 %[[SELECT]], ptr %[[RETVAL]]
// LLVM: %[[RESULT:.*]] = load i32, ptr %[[RETVAL]]
// LLVM: ret i32 %[[RESULT]]

// OGCG-LABEL: define{{.*}} i32 @_Z1xi(
// OGCG-SAME: i32 {{.*}} %[[ARG0:.+]])
// OGCG: %[[Y:.*]] = alloca i32
// OGCG: store i32 %[[ARG0]], ptr %[[Y]]
// OGCG: %[[YVAL:.*]] = load i32, ptr %[[Y]]
// OGCG: %[[CMP:.*]] = icmp sgt i32 %[[YVAL]], 0
// OGCG: %[[SELECT:.*]] = select i1 %[[CMP]], i32 3, i32 5
// OGCG: ret i32 %[[SELECT]]

int foo(int a, int b) {
  if (a < b ? 0 : a)
    return -1;
  return 0;
}

// CIR-LABEL: cir.func{{.*}} @_Z3fooii(
// CIR-SAME: %[[ARG0:.*]]: !s32i {{.*}}, %[[ARG1:.*]]: !s32i {{.*}}) -> !s32i{{.*}}{
// CIR: [[A:%.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: [[B:%.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: [[RETVAL:%.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: cir.store %[[ARG0]], [[A]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store %[[ARG1]], [[B]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.scope {
// CIR: [[ALOAD:%.+]] = cir.load{{.*}} [[A]] : !cir.ptr<!s32i>, !s32i
// CIR: [[BLOAD:%.+]] = cir.load{{.*}} [[B]] : !cir.ptr<!s32i>, !s32i
// CIR: [[CMP:%.+]] = cir.cmp(lt, [[ALOAD]], [[BLOAD]]) : !s32i, !cir.bool
// CIR: [[TERNARY_RES:%.+]] = cir.ternary([[CMP]], true {
// CIR: [[ZERO:%.+]] = cir.const #cir.int<0> : !s32i
// CIR: cir.yield [[ZERO]] : !s32i
// CIR: }, false {
// CIR: [[ALOAD2:%.+]] = cir.load{{.*}} [[A]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.yield [[ALOAD2]] : !s32i
// CIR: }) : (!cir.bool) -> !s32i
// CIR: [[CAST:%.+]] = cir.cast int_to_bool [[TERNARY_RES]] : !s32i -> !cir.bool
// CIR: cir.if [[CAST]] {
// CIR: [[ONE:%.+]] = cir.const #cir.int<1> : !s32i
// CIR: [[MINUS_ONE:%.+]] = cir.unary(minus, [[ONE]]) nsw : !s32i, !s32i
// CIR: cir.store [[MINUS_ONE]], [[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR: [[RETVAL_VAL:%.+]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return [[RETVAL_VAL]] : !s32i
// CIR: }
// CIR: }
// CIR: [[ZERO2:%.+]] = cir.const #cir.int<0> : !s32i
// CIR: cir.store [[ZERO2]], [[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR: [[RETVAL_VAL2:%.+]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return [[RETVAL_VAL2]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z3fooii(
// LLVM-SAME: i32 %[[ARG0:.*]], i32 %[[ARG1:.*]])
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM: %[[RETVAL:.*]] = alloca i32
// LLVM: store i32 %[[ARG0]], ptr %[[A]]
// LLVM: store i32 %[[ARG1]], ptr %[[B]]
// LLVM: br label %[[ENTRY_BB:.*]]
// LLVM: [[ENTRY_BB]]:
// LLVM: %[[AVAL:.*]] = load i32, ptr %[[A]]
// LLVM: %[[BVAL:.*]] = load i32, ptr %[[B]]
// LLVM: %[[CMP:.*]] = icmp slt i32 %[[AVAL]], %[[BVAL]]
// LLVM: br i1 %[[CMP]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM: br label %[[MERGE_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM: %[[AVAL2:.*]] = load i32, ptr %[[A]]
// LLVM: br label %[[MERGE_BB]]
// LLVM: [[MERGE_BB]]:
// LLVM: %[[PHI:.*]] = phi i32 [ %[[AVAL2]], %[[FALSE_BB]] ], [ 0, %[[TRUE_BB]] ]
// LLVM: %[[COND:.*]] = icmp ne i32 %[[PHI]], 0
// LLVM: br i1 %[[COND]], label %[[RETURN_MINUS_ONE:.*]], label %[[CONT_BB:.*]]
// LLVM: [[RETURN_MINUS_ONE]]:
// LLVM: store i32 -1, ptr %[[RETVAL]]
// LLVM: %[[RET1:.*]] = load i32, ptr %[[RETVAL]]
// LLVM: ret i32 %[[RET1]]
// LLVM: [[CONT_BB]]:
// LLVM: br label %[[RETURN_ZERO:.*]]
// LLVM: [[RETURN_ZERO]]:
// LLVM: store i32 0, ptr %[[RETVAL]]
// LLVM: %[[RET2:.*]] = load i32, ptr %[[RETVAL]]
// LLVM: ret i32 %[[RET2]]

// OGCG-LABEL: define{{.*}} i32 @_Z3fooii(
// OGCG-SAME: i32 {{.*}} %[[ARG0:.*]], i32 {{.*}} %[[ARG1:.*]])
// OGCG: %[[RETVAL:.*]] = alloca i32
// OGCG: %[[A:.*]] = alloca i32
// OGCG: %[[B:.*]] = alloca i32
// OGCG: store i32 %[[ARG0]], ptr %[[A]]
// OGCG: store i32 %[[ARG1]], ptr %[[B]]
// OGCG: %[[AVAL:.*]] = load i32, ptr %[[A]]
// OGCG: %[[BVAL:.*]] = load i32, ptr %[[B]]
// OGCG: %[[CMP:.*]] = icmp slt i32 %[[AVAL]], %[[BVAL]]
// OGCG: br i1 %[[CMP]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG: br label %[[MERGE_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG: %[[AVAL2:.*]] = load i32, ptr %[[A]]
// OGCG: br label %[[MERGE_BB]]
// OGCG: [[MERGE_BB]]:
// OGCG: %[[PHI:.*]] = phi i32 [ 0, %[[TRUE_BB]] ], [ %[[AVAL2]], %[[FALSE_BB]] ]
// OGCG: %[[COND:.*]] = icmp ne i32 %[[PHI]], 0
// OGCG: br i1 %[[COND]], label %[[RETURN_MINUS_ONE:.*]], label %[[RETURN_ZERO:.*]]
// OGCG: [[RETURN_MINUS_ONE]]:
// OGCG: store i32 -1, ptr %[[RETVAL]]
// OGCG: br label %[[RETURN:.+]]
// OGCG: [[RETURN_ZERO]]:
// OGCG: store i32 0, ptr %[[RETVAL]]
// OGCG: br label %[[RETURN]]
// OGCG: [[RETURN]]:
// OGCG: %[[RET2:.*]] = load i32, ptr %[[RETVAL]]
// OGCG: ret i32 %[[RET2]]

void test_cond_lvalue_assign(bool flag) {
  int a = 1;
  int b = 2;
  (flag ? a : b) = 42;
}

// CIR-LABEL: cir.func{{.*}} @_Z23test_cond_lvalue_assignb(
// CIR: %[[FLAG:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["flag", init]
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]]
// CIR: %[[TERNARY_PTR:.*]] = cir.ternary(%[[FLAG_VAL]], true {
// CIR:   cir.yield %[[A]] : !cir.ptr<!s32i>
// CIR: }, false {
// CIR:   cir.yield %[[B]] : !cir.ptr<!s32i>
// CIR: }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %{{.*}}, %[[TERNARY_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} void @_Z23test_cond_lvalue_assignb(
// LLVM: %[[FLAG:.*]] = alloca i8
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM: %[[FLAG_VAL:.*]] = load i8, ptr %[[FLAG]]
// LLVM: %[[COND:.*]] = trunc i8 %[[FLAG_VAL]] to i1
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

// OGCG-LABEL: define{{.*}} void @_Z23test_cond_lvalue_assignb(
// OGCG: %[[FLAG:.*]] = alloca i8
// OGCG: %[[A:.*]] = alloca i32
// OGCG: %[[B:.*]] = alloca i32
// OGCG: %[[FLAG_VAL:.*]] = load i8, ptr %[[FLAG]]
// OGCG: %[[COND:.*]] = trunc i8 %[[FLAG_VAL]] to i1
// OGCG: br i1 %[[COND]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   br label %[[MERGE_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   br label %[[MERGE_BB]]
// OGCG: [[MERGE_BB]]:
// OGCG:   %[[PHI_PTR:.*]] = phi ptr [ %[[A]], %[[TRUE_BB]] ], [ %[[B]], %[[FALSE_BB]] ]
// OGCG:   store i32 42, ptr %[[PHI_PTR]]

int& test_cond_lvalue_ref(bool cond, int x, int y) {
  return cond ? x : y;
}

// CIR-LABEL: cir.func{{.*}} @_Z20test_cond_lvalue_refbii(
// CIR: %[[COND:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cond", init]
// CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR: %[[Y:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
// CIR: %[[COND_VAL:.*]] = cir.load{{.*}} %[[COND]]
// CIR: %[[REF_PTR:.*]] = cir.ternary(%[[COND_VAL]], true {
// CIR:   cir.yield %[[X]] : !cir.ptr<!s32i>
// CIR: }, false {
// CIR:   cir.yield %[[Y]] : !cir.ptr<!s32i>
// CIR: }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR: cir.store %[[REF_PTR]]
// CIR: %[[RET_VAL:.*]] = cir.load
// CIR: cir.return %[[RET_VAL]] : !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} ptr @_Z20test_cond_lvalue_refbii(
// LLVM: %[[COND:.*]] = alloca i8
// LLVM: %[[X:.*]] = alloca i32
// LLVM: %[[Y:.*]] = alloca i32
// LLVM: %[[COND_VAL:.*]] = load i8, ptr %[[COND]]
// LLVM: %[[BOOL:.*]] = trunc i8 %[[COND_VAL]] to i1
// LLVM: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   br label %[[MERGE_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   br label %[[MERGE_BB]]
// LLVM: [[MERGE_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[Y]], %[[FALSE_BB]] ], [ %[[X]], %[[TRUE_BB]] ]
// LLVM:   br label %[[CONT:.*]]
// LLVM: [[CONT]]:
// LLVM:   store ptr %[[PHI]]
// LLVM:   %[[RET:.*]] = load ptr
// LLVM:   ret ptr %[[RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z20test_cond_lvalue_refbii(
// OGCG: %[[COND:.*]] = alloca i8
// OGCG: %[[X:.*]] = alloca i32
// OGCG: %[[Y:.*]] = alloca i32
// OGCG: %[[COND_VAL:.*]] = load i8, ptr %[[COND]]
// OGCG: %[[BOOL:.*]] = trunc i8 %[[COND_VAL]] to i1
// OGCG: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   br label %[[MERGE_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   br label %[[MERGE_BB]]
// OGCG: [[MERGE_BB]]:
// OGCG:   %[[PHI:.*]] = phi ptr [ %[[X]], %[[TRUE_BB]] ], [ %[[Y]], %[[FALSE_BB]] ]
// OGCG:   ret ptr %[[PHI]]

// Test ConditionalOperator as lvalue - compound assignment
void test_cond_lvalue_compound(bool flag) {
  int a = 5;
  int b = 10;
  (flag ? a : b) += 3;
}

// CIR-LABEL: cir.func{{.*}} @_Z25test_cond_lvalue_compoundb(
// CIR: %[[FLAG:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["flag", init]
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]]
// CIR: %[[LVAL_PTR:.*]] = cir.ternary(%[[FLAG_VAL]], true {
// CIR:   cir.yield %[[A]] : !cir.ptr<!s32i>
// CIR: }, false {
// CIR:   cir.yield %[[B]] : !cir.ptr<!s32i>
// CIR: }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR: %[[OLD_VAL:.*]] = cir.load{{.*}} %[[LVAL_PTR]]
// CIR: %[[NEW_VAL:.*]] = cir.binop(add, %[[OLD_VAL]], %{{.*}})
// CIR: cir.store{{.*}} %[[NEW_VAL]], %[[LVAL_PTR]]

// LLVM-LABEL: define{{.*}} void @_Z25test_cond_lvalue_compoundb(
// LLVM: %[[FLAG:.*]] = alloca i8
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM: %[[FLAG_VAL:.*]] = load i8, ptr %[[FLAG]]
// LLVM: %[[BOOL:.*]] = trunc i8 %[[FLAG_VAL]] to i1
// LLVM: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   br label %[[MERGE_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   br label %[[MERGE_BB]]
// LLVM: [[MERGE_BB]]:
// LLVM:   %[[PTR:.*]] = phi ptr [ %[[B]], %[[FALSE_BB]] ], [ %[[A]], %[[TRUE_BB]] ]
// LLVM:   br label %[[CONT:.*]]
// LLVM: [[CONT]]:
// LLVM:   %[[OLD:.*]] = load i32, ptr %[[PTR]]
// LLVM:   %[[NEW:.*]] = add{{.*}} i32 %[[OLD]], 3
// LLVM:   store i32 %[[NEW]], ptr %[[PTR]]

// OGCG-LABEL: define{{.*}} void @_Z25test_cond_lvalue_compoundb(
// OGCG: %[[FLAG:.*]] = alloca i8
// OGCG: %[[A:.*]] = alloca i32
// OGCG: %[[B:.*]] = alloca i32
// OGCG: %[[FLAG_VAL:.*]] = load i8, ptr %[[FLAG]]
// OGCG: %[[BOOL:.*]] = trunc i8 %[[FLAG_VAL]] to i1
// OGCG: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   br label %[[MERGE_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   br label %[[MERGE_BB]]
// OGCG: [[MERGE_BB]]:
// OGCG:   %[[PTR:.*]] = phi ptr [ %[[A]], %[[TRUE_BB]] ], [ %[[B]], %[[FALSE_BB]] ]
// OGCG:   %[[OLD:.*]] = load i32, ptr %[[PTR]]
// OGCG:   %[[NEW:.*]] = add{{.*}} i32 %[[OLD]], 3
// OGCG:   store i32 %[[NEW]], ptr %[[PTR]]

// Test constant folding - compile-time true condition with lvalue assignment
void test_cond_const_true_lvalue() {
  int a = 1;
  int b = 2;
  (true ? a : b) = 99;
}

// CIR-LABEL: cir.func{{.*}} @_Z27test_cond_const_true_lvaluev(
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR-NOT: cir.ternary
// CIR: %[[NINETYNINE:.*]] = cir.const #cir.int<99> : !s32i
// CIR: cir.store{{.*}} %[[NINETYNINE]], %[[A]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} void @_Z27test_cond_const_true_lvaluev(
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM-NOT: br i1
// LLVM: store i32 99, ptr %[[A]]

// OGCG-LABEL: define{{.*}} void @_Z27test_cond_const_true_lvaluev(
// OGCG: %[[A:.*]] = alloca i32
// OGCG: %[[B:.*]] = alloca i32
// OGCG-NOT: br i1
// OGCG: store i32 99, ptr %[[A]]

// Test constant folding - compile-time false condition with lvalue assignment
void test_cond_const_false_lvalue() {
  int a = 1;
  int b = 2;
  (false ? a : b) = 88;
}

// CIR-LABEL: cir.func{{.*}} @_Z28test_cond_const_false_lvaluev(
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR-NOT: cir.ternary
// CIR: %[[EIGHTYEIGHT:.*]] = cir.const #cir.int<88> : !s32i
// CIR: cir.store{{.*}} %[[EIGHTYEIGHT]], %[[B]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} void @_Z28test_cond_const_false_lvaluev(
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM-NOT: br i1
// LLVM: store i32 88, ptr %[[B]]

// OGCG-LABEL: define{{.*}} void @_Z28test_cond_const_false_lvaluev(
// OGCG: %[[A:.*]] = alloca i32
// OGCG: %[[B:.*]] = alloca i32
// OGCG-NOT: br i1
// OGCG: store i32 88, ptr %[[B]]
