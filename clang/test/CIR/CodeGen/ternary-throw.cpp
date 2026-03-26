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

const int &test_cond_const_true_throw_true() {
  const int a = 30;
  return true ? throw 0 : a;
}

// CIR-LABEL: cir.func{{.*}} @_Z31test_cond_const_true_throw_truev(
// CIR:  %[[RET_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"]
// CIR:  %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init, const]
// CIR:  %[[CONST_30:.*]] = cir.const #cir.int<30> : !s32i
// CIR:  cir.store{{.*}} %[[CONST_30]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:  %[[EXCEPTION:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR:  %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR:  cir.store{{.*}} %[[CONST_0]], %[[EXCEPTION]] : !s32i, !cir.ptr<!s32i>
// CIR:  cir.throw %[[EXCEPTION]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:  cir.unreachable
// CIR: ^[[NO_PRED_LABEL:.*]]:
// CIR:   %[[CONST_NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR:   cir.store %[[CONST_NULL]], %[[RET_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   cir.return %[[TMP_RET]] : !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} ptr @_Z31test_cond_const_true_throw_truev(
// LLVM:  %[[RET_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:  %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:  store i32 30, ptr %[[A_ADDR]], align 4
// LLVM:  %[[EXCEPTION:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// LLVM:  store i32 0, ptr %[[EXCEPTION]], align 16
// LLVM:  call void @__cxa_throw(ptr %[[EXCEPTION]], ptr @_ZTIi, ptr null)
// LLVM:  unreachable
// LLVM: [[NO_PRED_LABEL:.*]]:
// LLVM:  store ptr null, ptr %[[RET_ADDR]], align 8
// LLVM:  %[[TMP_RET:.*]] = load ptr, ptr %[[RET_ADDR]], align 8
// LLVM:  ret ptr %[[TMP_RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z31test_cond_const_true_throw_truev(
// OGCG:  %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:  store i32 30, ptr %[[A_ADDR]], align 4
// OGCG:  %[[EXCEPTION:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// OGCG:  store i32 0, ptr %[[EXCEPTION]], align 16
// OGCG:  call void @__cxa_throw(ptr %[[EXCEPTION]], ptr @_ZTIi, ptr null)
// OGCG:  unreachable
// OGCG: [[NO_PRED_LABEL:.*]]:
// OGCG:  ret ptr [[UNDEF:.*]]

struct s6 { int f0; };
int test_agg_cond_throw_false(bool flag, struct s6 a1, struct s6 a2) {
  return (flag ? a1 : throw 0).f0;
}

// CIR-LABEL: cir.func{{.*}} @_Z25test_agg_cond_throw_falseb2s6S_(
// CIR: %[[FLAG:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["flag", init]
// CIR: %[[A1:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a1", init]
// CIR: %[[A2:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a2", init]
// CIR: %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[COND_RES:.*]] = cir.ternary(%[[FLAG_VAL]], true {
// CIR:   cir.yield %[[A1]] : !cir.ptr<!rec_s6>
// CIR: }, false {
// CIR:   %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[EXC]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:   cir.unreachable
// CIR: }) : (!cir.bool) -> !cir.ptr<!rec_s6>
// CIR: %[[F0:.*]] = cir.get_member %[[A1]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
// CIR: %[[LOAD:.*]] = cir.load{{.*}} %[[F0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %{{.*}} : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z25test_agg_cond_throw_falseb2s6S_(
// LLVM: %[[FLAG_ALLOCA:.*]] = alloca i8
// LLVM: %[[A1_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[A2_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[ZEXT:.*]] = zext i1 %{{.*}} to i8
// LLVM: store i8 %[[ZEXT]], ptr %[[FLAG_ALLOCA]]
// LLVM: %[[FLAG_LOAD:.*]] = load i8, ptr %[[FLAG_ALLOCA]]
// LLVM: %[[BOOL:.*]] = trunc i8 %[[FLAG_LOAD]] to i1
// LLVM: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   br label %[[PHI_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// LLVM:   store i32 0, ptr %[[EXC]]
// LLVM:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[PHI_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[A1_ALLOCA]], %[[TRUE_BB]] ]
// LLVM:   br label %[[CONT_BB:.*]]
// LLVM: [[CONT_BB]]:
// LLVM:   %[[F0_PTR:.*]] = getelementptr %struct.s6, ptr %[[A1_ALLOCA]], i32 0, i32 0
// LLVM:   %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// LLVM:   ret i32 %{{.*}}

// OGCG-LABEL: define{{.*}} i32 @_Z25test_agg_cond_throw_falseb2s6S_(
// OGCG: %[[A1:.*]] = alloca %struct.s6
// OGCG: %[[A2:.*]] = alloca %struct.s6
// OGCG: %{{.*}} = alloca i8
// OGCG: %[[LOAD:.*]] = load i8, ptr %{{.*}}
// OGCG: %[[BOOL:.*]] = trunc i8 %[[LOAD]] to i1
// OGCG: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   br label %[[END:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// OGCG:   store i32 0, ptr %[[EXC]]
// OGCG:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[END]]:
// OGCG:   %[[F0_PTR:.*]] = getelementptr inbounds nuw %struct.s6, ptr %[[A1]], i32 0, i32 0
// OGCG:   %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// OGCG:   ret i32 %{{.*}}

int test_agg_cond_throw_true(bool flag, struct s6 a1, struct s6 a2) {
  return (flag ? throw 0 : a1).f0;
}

// CIR-LABEL: cir.func{{.*}} @_Z24test_agg_cond_throw_trueb2s6S_(
// CIR: %[[FLAG:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["flag", init]
// CIR: %[[A1:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a1", init]
// CIR: %[[A2:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a2", init]
// CIR: %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[COND_RES:.*]] = cir.ternary(%[[FLAG_VAL]], true {
// CIR:   %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[EXC]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:   cir.unreachable
// CIR: }, false {
// CIR:   cir.yield %[[A1]] : !cir.ptr<!rec_s6>
// CIR: }) : (!cir.bool) -> !cir.ptr<!rec_s6>
// CIR: %[[F0:.*]] = cir.get_member %[[A1]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
// CIR: %[[LOAD:.*]] = cir.load{{.*}} %[[F0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %{{.*}} : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z24test_agg_cond_throw_trueb2s6S_(
// LLVM: %[[FLAG_ALLOCA:.*]] = alloca i8
// LLVM: %[[A1_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[A2_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[ZEXT:.*]] = zext i1 %{{.*}} to i8
// LLVM: store i8 %[[ZEXT]], ptr %[[FLAG_ALLOCA]]
// LLVM: %[[FLAG_LOAD:.*]] = load i8, ptr %[[FLAG_ALLOCA]]
// LLVM: %[[BOOL:.*]] = trunc i8 %[[FLAG_LOAD]] to i1
// LLVM: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// LLVM:   store i32 0, ptr %[[EXC]]
// LLVM:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   br label %[[PHI_BB:.*]]
// LLVM: [[PHI_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[A1_ALLOCA]], %[[FALSE_BB]] ]
// LLVM:   br label %[[CONT_BB:.*]]
// LLVM: [[CONT_BB]]:
// LLVM:   %[[F0_PTR:.*]] = getelementptr %struct.s6, ptr %[[A1_ALLOCA]], i32 0, i32 0
// LLVM:   %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// LLVM:   ret i32 %{{.*}}

// OGCG-LABEL: define{{.*}} i32 @_Z24test_agg_cond_throw_trueb2s6S_(
// OGCG: %[[A1:.*]] = alloca %struct.s6
// OGCG: %[[A2:.*]] = alloca %struct.s6
// OGCG: %{{.*}} = alloca i8
// OGCG: %[[LOAD:.*]] = load i8, ptr %{{.*}}
// OGCG: %[[BOOL:.*]] = trunc i8 %[[LOAD]] to i1
// OGCG: br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// OGCG:   store i32 0, ptr %[[EXC]]
// OGCG:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   br label %[[END:.*]]
// OGCG: [[END]]:
// OGCG:   %[[F0_PTR:.*]] = getelementptr inbounds nuw %struct.s6, ptr %[[A1]], i32 0, i32 0
// OGCG:   %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// OGCG:   ret i32 %{{.*}}

const int test_agg_cond_const_true_throw_false(struct s6 a1, struct s6 a2) {
  return (true ? a1 : throw 0).f0;
}

// CIR-LABEL: cir.func{{.*}} @_Z36test_agg_cond_const_true_throw_false2s6S_(
// CIR: %[[A1:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a1", init]
// CIR: %[[A2:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a2", init]
// CIR-NOT: cir.ternary
// CIR-NOT: cir.throw
// CIR: %[[F0:.*]] = cir.get_member %[[A1]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
// CIR: %[[LOAD:.*]] = cir.load{{.*}} %[[F0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %{{.*}} : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z36test_agg_cond_const_true_throw_false2s6S_(
// LLVM: %[[A1_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[A2_ALLOCA:.*]] = alloca %struct.s6
// LLVM-NOT: br i1
// LLVM-NOT: __cxa_throw
// LLVM: %[[F0_PTR:.*]] = getelementptr %struct.s6, ptr %[[A1_ALLOCA]], i32 0, i32 0
// LLVM: %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// LLVM: ret i32 %{{.*}}

// OGCG-LABEL: define{{.*}} i32 @_Z36test_agg_cond_const_true_throw_false2s6S_(
// OGCG: %[[A1:.*]] = alloca %struct.s6
// OGCG: %[[A2:.*]] = alloca %struct.s6
// Match the coerce stores so we match the F0 gep and load correctly.
// OGCG: getelementptr
// OGCG: store
// OGCG: getelementptr
// OGCG: store
// OGCG-NOT: br i1
// OGCG-NOT: __cxa_throw
// OGCG: %[[F0_PTR:.*]] = getelementptr inbounds nuw %struct.s6, ptr %[[A1]], i32 0, i32 0
// OGCG: %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// OGCG: ret i32 %{{.*}}

const int test_agg_cond_const_true_throw_true(struct s6 a1, struct s6 a2) {
  return (true ? throw 0 : a2).f0;
}

// CIR-LABEL: cir.func{{.*}} @_Z35test_agg_cond_const_true_throw_true2s6S_(
// CIR: %[[A1:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a1", init]
// CIR: %[[A2:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a2", init]
// CIR: %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR: cir.store{{.*}} %[[ZERO]], %[[EXC]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR: cir.unreachable
// CIR: ^[[NO_PRED:.*]]:
// CIR: %[[NULL_REC:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_s6>
// CIR: %[[F0:.*]] = cir.get_member %[[NULL_REC]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
// CIR: %[[LOAD:.*]] = cir.load{{.*}} %[[F0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %{{.*}} : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z35test_agg_cond_const_true_throw_true2s6S_(
// LLVM: %[[A1_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[A2_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// LLVM: store i32 0, ptr %[[EXC]]
// LLVM: call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// LLVM: unreachable
// LLVM: [[NO_PRED:.*]]:
// LLVM: %[[LOAD_UNDEF:.*]] = load i32, ptr null, align 1
// LLVM: ret i32 %{{.*}}

// OGCG-LABEL: define{{.*}} i32 @_Z35test_agg_cond_const_true_throw_true2s6S_(
// OGCG: %[[A1:.*]] = alloca %struct.s6
// OGCG: %[[A2:.*]] = alloca %struct.s6
// OGCG: %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// OGCG: store i32 0, ptr %[[EXC]]
// OGCG: call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// OGCG: unreachable
// OGCG: [[NO_PRED:.*]]:
// OGCG: %[[LOAD_UNDEF:.*]] = load i32, ptr undef, align 1
// OGCG: ret i32 %{{.*}}

const int test_agg_cond_const_false_throw_false(struct s6 a1, struct s6 a2) {
  return (false ? a1 : throw 0).f0;
}

// CIR-LABEL: cir.func{{.*}} @_Z37test_agg_cond_const_false_throw_false2s6S_(
// CIR: %[[A1:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a1", init]
// CIR: %[[A2:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a2", init]
// CIR: %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR: cir.store{{.*}} %[[ZERO]], %[[EXC]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR: cir.unreachable
// CIR: ^[[NO_PRED:.*]]:
// CIR: %[[NULL_REC:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_s6>
// CIR: %[[F0:.*]] = cir.get_member %[[NULL_REC]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
// CIR: %[[LOAD:.*]] = cir.load{{.*}} %[[F0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %{{.*}} : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z37test_agg_cond_const_false_throw_false2s6S_(
// LLVM: %[[A1_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[A2_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// LLVM: store i32 0, ptr %[[EXC]]
// LLVM: call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// LLVM: unreachable
// LLVM: [[NO_PRED:.*]]:
// LLVM: %[[LOAD_UNDEF:.*]] = load i32, ptr null, align 1
// LLVM: ret i32 %{{.*}}

// OGCG-LABEL: define{{.*}} i32 @_Z37test_agg_cond_const_false_throw_false2s6S_(
// OGCG: %[[A1:.*]] = alloca %struct.s6
// OGCG: %[[A2:.*]] = alloca %struct.s6
// OGCG: %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// OGCG: store i32 0, ptr %[[EXC]]
// OGCG: call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// OGCG: unreachable
// OGCG: [[NO_PRED:.*]]:
// OGCG: %[[LOAD_UNDEF:.*]] = load i32, ptr undef, align 1
// OGCG: ret i32 %{{.*}}

const int test_agg_cond_const_false_throw_true(struct s6 a1, struct s6 a2) {
  return (false ? throw 0 : a1).f0;
}

// CIR-LABEL: cir.func{{.*}} @_Z36test_agg_cond_const_false_throw_true2s6S_(
// CIR: %[[A1:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a1", init]
// CIR: %[[A2:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a2", init]
// CIR-NOT: cir.ternary
// CIR-NOT: cir.throw
// CIR: %[[F0:.*]] = cir.get_member %[[A1]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
// CIR: %[[LOAD:.*]] = cir.load{{.*}} %[[F0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %{{.*}} : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z36test_agg_cond_const_false_throw_true2s6S_(
// LLVM: %[[A1_ALLOCA:.*]] = alloca %struct.s6
// LLVM: %[[A2_ALLOCA:.*]] = alloca %struct.s6
// LLVM-NOT: br i1
// LLVM-NOT: __cxa_throw
// LLVM: %[[F0_PTR:.*]] = getelementptr %struct.s6, ptr %[[A1_ALLOCA]], i32 0, i32 0
// LLVM: %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// LLVM: ret i32 %{{.*}}

// OGCG-LABEL: define{{.*}} i32 @_Z36test_agg_cond_const_false_throw_true2s6S_(
// OGCG: %[[A1:.*]] = alloca %struct.s6
// OGCG: %[[A2:.*]] = alloca %struct.s6
// Match the coerce stores so we match the F0 gep and load correctly.
// OGCG: getelementptr
// OGCG: store
// OGCG: getelementptr
// OGCG: store
// OGCG-NOT: br i1
// OGCG-NOT: __cxa_throw
// OGCG: %[[F0_PTR:.*]] = getelementptr inbounds nuw %struct.s6, ptr %[[A1]], i32 0, i32 0
// OGCG: %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// OGCG: ret i32 %{{.*}}
