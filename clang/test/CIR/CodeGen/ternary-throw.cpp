// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexceptions -fcxx-exceptions -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexceptions -fcxx-exceptions -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

const int& test_cond_throw_false(bool flag) {
  const int a = 10;
  return flag ? a : throw 0;
}

// CIR-LABEL: cir.func{{.*}} @_Z21test_cond_throw_falseb(
// CIR: %[[FLAG:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["flag", init]
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init, const]
// CIR: %[[TEN:.*]] = cir.const #cir.int<10> : !s32i
// CIR: cir.store{{.*}} %[[TEN]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[RESULT:.*]] = cir.ternary(%[[FLAG_VAL]], true {
// CIR:   cir.yield %[[A]] : !cir.ptr<!s32i>
// CIR: }, false {
// CIR:   %[[EXCEPTION:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[EXCEPTION]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.throw %[[EXCEPTION]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:   cir.unreachable
// CIR: }) : (!cir.bool) -> !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} ptr @_Z21test_cond_throw_falseb(
// LLVM: %[[FLAG_ALLOCA:.*]] = alloca i8
// LLVM: %[[RET_ALLOCA:.*]] = alloca ptr
// LLVM: %[[A_ALLOCA:.*]] = alloca i32
// LLVM: %[[ZEXT:.*]] = zext i1 %{{.*}} to i8
// LLVM: store i8 %[[ZEXT]], ptr %[[FLAG_ALLOCA]]
// LLVM: store i32 10, ptr %[[A_ALLOCA]]
// LLVM: %[[LOAD:.*]] = load i8, ptr %[[FLAG_ALLOCA]]
// LLVM: %[[BOOL:.*]] = trunc i8 %[[LOAD]] to i1
// LLVM: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   br label %[[PHI_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// LLVM:   store i32 0, ptr %[[EXC]]
// LLVM:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[PHI_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[A_ALLOCA]], %[[TRUE_BB]] ]
// LLVM:   br label %[[CONT_BB:.*]]
// LLVM: [[CONT_BB]]:
// LLVM:   store ptr %[[A_ALLOCA]], ptr %[[RET_ALLOCA]]
// LLVM:   %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM:   ret ptr %[[RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z21test_cond_throw_falseb(
// OGCG: %{{.*}} = alloca i8
// OGCG: %[[A:.*]] = alloca i32
// OGCG: store i32 10, ptr %[[A]]
// OGCG: %{{.*}} = load i8, ptr %{{.*}}
// OGCG: %[[BOOL:.*]] = trunc i8 %{{.*}} to i1
// OGCG: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   br label %[[END:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   %{{.*}} = call{{.*}} ptr @__cxa_allocate_exception
// OGCG:   store i32 0, ptr %{{.*}}
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[END]]:
// OGCG:   ret ptr %[[A]]

const int& test_cond_throw_true(bool flag) {
  const int a = 10;
  return flag ? throw 0 : a;
}

// CIR-LABEL: cir.func{{.*}} @_Z20test_cond_throw_trueb(
// CIR: %[[FLAG:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["flag", init]
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init, const]
// CIR: %[[TEN:.*]] = cir.const #cir.int<10> : !s32i
// CIR: cir.store{{.*}} %[[TEN]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[RESULT:.*]] = cir.ternary(%[[FLAG_VAL]], true {
// CIR:   %[[EXCEPTION:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[EXCEPTION]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.throw %[[EXCEPTION]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:   cir.unreachable
// CIR: }, false {
// CIR:   cir.yield %[[A]] : !cir.ptr<!s32i>
// CIR: }) : (!cir.bool) -> !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} ptr @_Z20test_cond_throw_trueb(
// LLVM: %[[FLAG_ALLOCA:.*]] = alloca i8
// LLVM: %[[RET_ALLOCA:.*]] = alloca ptr
// LLVM: %[[A_ALLOCA:.*]] = alloca i32
// LLVM: %[[ZEXT:.*]] = zext i1 %{{.*}} to i8
// LLVM: store i8 %[[ZEXT]], ptr %[[FLAG_ALLOCA]]
// LLVM: store i32 10, ptr %[[A_ALLOCA]]
// LLVM: %[[LOAD:.*]] = load i8, ptr %[[FLAG_ALLOCA]]
// LLVM: %[[BOOL:.*]] = trunc i8 %[[LOAD]] to i1
// LLVM: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// LLVM:   store i32 0, ptr %[[EXC]]
// LLVM:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   br label %[[PHI_BB:.*]]
// LLVM: [[PHI_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[A_ALLOCA]], %[[FALSE_BB]] ]
// LLVM:   br label %[[CONT_BB:.*]]
// LLVM: [[CONT_BB]]:
// LLVM:   store ptr %[[A_ALLOCA]], ptr %[[RET_ALLOCA]]
// LLVM:   %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM:   ret ptr %[[RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z20test_cond_throw_trueb(
// OGCG: %{{.*}} = alloca i8
// OGCG: %[[A:.*]] = alloca i32
// OGCG: store i32 10, ptr %[[A]]
// OGCG: %{{.*}} = load i8, ptr %{{.*}}
// OGCG: %[[BOOL:.*]] = trunc i8 %{{.*}} to i1
// OGCG: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   %{{.*}} = call{{.*}} ptr @__cxa_allocate_exception
// OGCG:   store i32 0, ptr %{{.*}}
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   br label %[[END:.*]]
// OGCG: [[END]]:
// OGCG:   ret ptr %[[A]]

// Test constant folding with throw - compile-time true condition, dead throw in false branch
const int& test_cond_const_true_throw_false() {
  const int a = 20;
  return true ? a : throw 0;
}

// CIR-LABEL: cir.func{{.*}} @_Z32test_cond_const_true_throw_falsev(
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init, const]
// CIR: %[[TWENTY:.*]] = cir.const #cir.int<20> : !s32i
// CIR: cir.store{{.*}} %[[TWENTY]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR-NOT: cir.ternary
// CIR-NOT: cir.throw
// CIR: cir.store %[[A]]
// CIR: %[[RET:.*]] = cir.load
// CIR: cir.return %[[RET]] : !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} ptr @_Z32test_cond_const_true_throw_falsev(
// LLVM: %[[A:.*]] = alloca i32
// LLVM: store i32 20, ptr %[[A]]
// LLVM-NOT: br i1
// LLVM-NOT: __cxa_throw
// LLVM: store ptr %[[A]]
// LLVM: %[[RET:.*]] = load ptr
// LLVM: ret ptr %[[RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z32test_cond_const_true_throw_falsev(
// OGCG: %[[A:.*]] = alloca i32
// OGCG: store i32 20, ptr %[[A]]
// OGCG-NOT: br i1
// OGCG-NOT: __cxa_throw
// OGCG: ret ptr %[[A]]

// Test constant folding with throw - compile-time false condition, dead throw in true branch
const int& test_cond_const_false_throw_true() {
  const int a = 30;
  return false ? throw 0 : a;
}

// CIR-LABEL: cir.func{{.*}} @_Z32test_cond_const_false_throw_truev(
// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init, const]
// CIR: %[[THIRTY:.*]] = cir.const #cir.int<30> : !s32i
// CIR: cir.store{{.*}} %[[THIRTY]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR-NOT: cir.ternary
// CIR-NOT: cir.throw
// CIR: cir.store %[[A]]
// CIR: %[[RET:.*]] = cir.load
// CIR: cir.return %[[RET]] : !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} ptr @_Z32test_cond_const_false_throw_truev(
// LLVM: %[[A:.*]] = alloca i32
// LLVM: store i32 30, ptr %[[A]]
// LLVM-NOT: br i1
// LLVM-NOT: __cxa_throw
// LLVM: store ptr %[[A]]
// LLVM: %[[RET:.*]] = load ptr
// LLVM: ret ptr %[[RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z32test_cond_const_false_throw_truev(
// OGCG: %[[A:.*]] = alloca i32
// OGCG: store i32 30, ptr %[[A]]
// OGCG-NOT: br i1
// OGCG-NOT: __cxa_throw
// OGCG: ret ptr %[[A]]

