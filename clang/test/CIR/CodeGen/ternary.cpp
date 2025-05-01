// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

int x(int y) {
  return y > 0 ? 3 : 5;
}

// CIR-LABEL: cir.func @_Z1xi(
// CIR-SAME: %[[ARG0:.*]]: !s32i {{.*}}) -> !s32i {
// CIR: [[Y:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
// CIR: [[RETVAL:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: cir.store %[[ARG0]], [[Y]] : !s32i, !cir.ptr<!s32i>
// CIR: [[YVAL:%[0-9]+]] = cir.load [[Y]] : !cir.ptr<!s32i>, !s32i
// CIR: [[ZERO:%[0-9]+]] = cir.const #cir.int<0> : !s32i
// CIR: [[CMP:%[0-9]+]] = cir.cmp(gt, [[YVAL]], [[ZERO]]) : !s32i, !cir.bool
// CIR: [[TERNARY_RES:%[0-9]+]] = cir.ternary([[CMP]], true {
// CIR: [[THREE:%[0-9]+]] = cir.const #cir.int<3> : !s32i
// CIR: cir.yield [[THREE]] : !s32i
// CIR: }, false {
// CIR: [[FIVE:%[0-9]+]] = cir.const #cir.int<5> : !s32i
// CIR: cir.yield [[FIVE]] : !s32i
// CIR: }) : (!cir.bool) -> !s32i
// CIR: cir.store [[TERNARY_RES]], [[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR: [[RETVAL_VAL:%[0-9]+]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return [[RETVAL_VAL]] : !s32i

// LLVM-LABEL: define i32 @_Z1xi(
// LLVM-SAME: i32 %[[ARG0:.*]])
// LLVM: %[[Y:.*]] = alloca i32
// LLVM: %[[RETVAL:.*]] = alloca i32
// LLVM: store i32 %[[ARG0]], ptr %[[Y]]
// LLVM: %[[YVAL:.*]] = load i32, ptr %[[Y]]
// LLVM: %[[CMP:.*]] = icmp sgt i32 %[[YVAL]], 0
// LLVM: br i1 %[[CMP]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM: br label %[[MERGE_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM: br label %[[MERGE_BB]]
// LLVM: [[MERGE_BB]]:
// LLVM: %[[PHI:.*]] = phi i32 [ 5, %[[FALSE_BB]] ], [ 3, %[[TRUE_BB]] ]
// LLVM: br label %[[FINAL_BB:.*]]
// LLVM: [[FINAL_BB]]:
// LLVM: store i32 %[[PHI]], ptr %[[RETVAL]]
// LLVM: %[[RESULT:.*]] = load i32, ptr %[[RETVAL]]
// LLVM: ret i32 %[[RESULT]]

int foo(int a, int b) {
  if (a < b ? 0 : a)
    return -1;
  return 0;
}

// CIR-LABEL: cir.func @_Z3fooii(
// CIR-SAME: %[[ARG0:.*]]: !s32i {{.*}}, %[[ARG1:.*]]: !s32i {{.*}}) -> !s32i {
// CIR: [[A:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: [[B:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: [[RETVAL:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: cir.store %[[ARG0]], [[A]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store %[[ARG1]], [[B]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.scope {
// CIR: [[ALOAD:%[0-9]+]] = cir.load [[A]] : !cir.ptr<!s32i>, !s32i
// CIR: [[BLOAD:%[0-9]+]] = cir.load [[B]] : !cir.ptr<!s32i>, !s32i
// CIR: [[CMP:%[0-9]+]] = cir.cmp(lt, [[ALOAD]], [[BLOAD]]) : !s32i, !cir.bool
// CIR: [[TERNARY_RES:%[0-9]+]] = cir.ternary([[CMP]], true {
// CIR: [[ZERO:%[0-9]+]] = cir.const #cir.int<0> : !s32i
// CIR: cir.yield [[ZERO]] : !s32i
// CIR: }, false {
// CIR: [[ALOAD2:%[0-9]+]] = cir.load [[A]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.yield [[ALOAD2]] : !s32i
// CIR: }) : (!cir.bool) -> !s32i
// CIR: [[CAST:%[0-9]+]] = cir.cast(int_to_bool, [[TERNARY_RES]] : !s32i), !cir.bool
// CIR: cir.if [[CAST]] {
// CIR: [[ONE:%[0-9]+]] = cir.const #cir.int<1> : !s32i
// CIR: [[MINUS_ONE:%[0-9]+]] = cir.unary(minus, [[ONE]]) nsw : !s32i, !s32i
// CIR: cir.store [[MINUS_ONE]], [[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR: [[RETVAL_VAL:%[0-9]+]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return [[RETVAL_VAL]] : !s32i
// CIR: }
// CIR: }
// CIR: [[ZERO2:%[0-9]+]] = cir.const #cir.int<0> : !s32i
// CIR: cir.store [[ZERO2]], [[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR: [[RETVAL_VAL2:%[0-9]+]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return [[RETVAL_VAL2]] : !s32i

// LLVM-LABEL: define i32 @_Z3fooii(
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
// LLVM: br label %[[CHECK_BB:.*]]
// LLVM: [[CHECK_BB]]:
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
