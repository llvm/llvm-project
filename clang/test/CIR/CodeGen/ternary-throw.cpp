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
// CIR: %[[FLAG:.*]] = cir.alloca "flag" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[A:.*]] = cir.alloca "a" {{.*}} init const : !cir.ptr<!s32i>
// CIR: %[[TEN:.*]] = cir.const #cir.int<10> : !s32i
// CIR: cir.store{{.*}} %[[TEN]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[RESULT:.*]] = cir.ternary(%[[FLAG_VAL]], true {
// CIR:   cir.yield %[[A]] : !cir.ptr<!s32i>
// CIR-NEXT: }, false {
// CIR:   %[[EXCEPTION:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[EXCEPTION]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.throw %[[EXCEPTION]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:   cir.unreachable
// CIR-NEXT: }) : (!cir.bool) -> !cir.ptr<!s32i>

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
// LLVM:   store ptr %[[PHI]], ptr %[[RET_ALLOCA]]
// LLVM:   %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM:   ret ptr %[[RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z21test_cond_throw_falseb(
// OGCG: %{{.*}} = alloca i8
// OGCG: %[[A:.*]] = alloca i32
// OGCG: store i32 10, ptr %[[A]]
// OGCG: %{{.*}} = load i8, ptr %{{.*}}
// OGCG: %[[BOOL:.*]] = icmp ne i8 %{{.*}}, 0
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
// CIR: %[[FLAG:.*]] = cir.alloca "flag" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[A:.*]] = cir.alloca "a" {{.*}} init const : !cir.ptr<!s32i>
// CIR: %[[TEN:.*]] = cir.const #cir.int<10> : !s32i
// CIR: cir.store{{.*}} %[[TEN]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[RESULT:.*]] = cir.ternary(%[[FLAG_VAL]], true {
// CIR:   %[[EXCEPTION:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[EXCEPTION]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.throw %[[EXCEPTION]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:   cir.unreachable
// CIR-NEXT: }, false {
// CIR:   cir.yield %[[A]] : !cir.ptr<!s32i>
// CIR-NEXT: }) : (!cir.bool) -> !cir.ptr<!s32i>

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
// LLVM:   store ptr %[[PHI]], ptr %[[RET_ALLOCA]]
// LLVM:   %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM:   ret ptr %[[RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z20test_cond_throw_trueb(
// OGCG: %{{.*}} = alloca i8
// OGCG: %[[A:.*]] = alloca i32
// OGCG: store i32 10, ptr %[[A]]
// OGCG: %{{.*}} = load i8, ptr %{{.*}}
// OGCG: %[[BOOL:.*]] = icmp ne i8 %{{.*}}, 0
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
// CIR: %[[A:.*]] = cir.alloca "a" {{.*}} init const : !cir.ptr<!s32i>
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
// CIR: %[[A:.*]] = cir.alloca "a" {{.*}} init const : !cir.ptr<!s32i>
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
// CIR:  %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.ptr<!s32i>>
// CIR:  %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init const : !cir.ptr<!s32i>
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
// CIR: %[[FLAG:.*]] = cir.alloca "flag" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[A1:.*]] = cir.alloca "a1" {{.*}} init : !cir.ptr<!rec_s6>
// CIR: %[[A2:.*]] = cir.alloca "a2" {{.*}} init : !cir.ptr<!rec_s6>
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
// CIR: %[[F0:.*]] = cir.get_member %[[COND_RES]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
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
// LLVM:   %[[F0_PTR:.*]] = getelementptr inbounds nuw %struct.s6, ptr %[[PHI]], i32 0, i32 0
// LLVM:   %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// LLVM:   ret i32 %{{.*}}

// OGCG-LABEL: define{{.*}} i32 @_Z25test_agg_cond_throw_falseb2s6S_(
// OGCG: %[[A1:.*]] = alloca %struct.s6
// OGCG: %[[A2:.*]] = alloca %struct.s6
// OGCG: %{{.*}} = alloca i8
// OGCG: %[[LOAD:.*]] = load i8, ptr %{{.*}}
// OGCG: %[[BOOL:.*]] = icmp ne i8 %[[LOAD]], 0
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
// CIR: %[[FLAG:.*]] = cir.alloca "flag" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[A1:.*]] = cir.alloca "a1" {{.*}} init : !cir.ptr<!rec_s6>
// CIR: %[[A2:.*]] = cir.alloca "a2" {{.*}} init : !cir.ptr<!rec_s6>
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
// CIR: %[[F0:.*]] = cir.get_member %[[COND_RES]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
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
// LLVM:   %[[F0_PTR:.*]] = getelementptr inbounds nuw %struct.s6, ptr %[[PHI]], i32 0, i32 0
// LLVM:   %[[F0_VAL:.*]] = load i32, ptr %[[F0_PTR]]
// LLVM:   ret i32 %{{.*}}

// OGCG-LABEL: define{{.*}} i32 @_Z24test_agg_cond_throw_trueb2s6S_(
// OGCG: %[[A1:.*]] = alloca %struct.s6
// OGCG: %[[A2:.*]] = alloca %struct.s6
// OGCG: %{{.*}} = alloca i8
// OGCG: %[[LOAD:.*]] = load i8, ptr %{{.*}}
// OGCG: %[[BOOL:.*]] = icmp ne i8 %[[LOAD]], 0
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
// CIR: %[[A1:.*]] = cir.alloca "a1" {{.*}} init : !cir.ptr<!rec_s6>
// CIR: %[[A2:.*]] = cir.alloca "a2" {{.*}} init : !cir.ptr<!rec_s6>
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
// LLVM: %[[F0_PTR:.*]] = getelementptr inbounds nuw %struct.s6, ptr %[[A1_ALLOCA]], i32 0, i32 0
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
// CIR: %[[A1:.*]] = cir.alloca "a1" {{.*}} init : !cir.ptr<!rec_s6>
// CIR: %[[A2:.*]] = cir.alloca "a2" {{.*}} init : !cir.ptr<!rec_s6>
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
// CIR: %[[A1:.*]] = cir.alloca "a1" {{.*}} init : !cir.ptr<!rec_s6>
// CIR: %[[A2:.*]] = cir.alloca "a2" {{.*}} init : !cir.ptr<!rec_s6>
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
// CIR: %[[A1:.*]] = cir.alloca "a1" {{.*}} init : !cir.ptr<!rec_s6>
// CIR: %[[A2:.*]] = cir.alloca "a2" {{.*}} init : !cir.ptr<!rec_s6>
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
// LLVM: %[[F0_PTR:.*]] = getelementptr inbounds nuw %struct.s6, ptr %[[A1_ALLOCA]], i32 0, i32 0
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

struct Agg {
  int x;
  int y;
};

void test_agg_throw_true(bool flag) {
  Agg a = flag ? throw 0 : Agg{1, 2};
}

// CIR-LABEL: cir.func{{.*}} @_Z19test_agg_throw_trueb(
// CIR:   %[[FLAG:.*]] = cir.alloca "flag"
// CIR:   %[[A:.*]] = cir.alloca "a"
// CIR:   %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   cir.if %[[FLAG_VAL]] {
// CIR:     %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:     cir.store{{.*}} %[[ZERO]], %[[EXC]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   } else {
// CIR:     %[[X:.*]] = cir.get_member %[[A]][0] {name = "x"} : !cir.ptr<!rec_Agg> -> !cir.ptr<!s32i>
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     cir.store{{.*}} %[[ONE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:     %[[Y:.*]] = cir.get_member %[[A]][1] {name = "y"} : !cir.ptr<!rec_Agg> -> !cir.ptr<!s32i>
// CIR:     %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:     cir.store{{.*}} %[[TWO]], %[[Y]] : !s32i, !cir.ptr<!s32i>
// CIR:   }
// CIR:   cir.return

// LLVM-LABEL: define{{.*}} void @_Z19test_agg_throw_trueb(
// LLVM:   %[[FLAG_ALLOCA:.*]] = alloca i8
// LLVM:   %[[A_ALLOCA:.*]] = alloca %struct.Agg
// LLVM:   %[[BOOL:.*]] = trunc i8 %{{.*}} to i1
// LLVM:   br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// LLVM:   store i32 0, ptr %[[EXC]]
// LLVM:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   %[[X:.*]] = getelementptr inbounds nuw %struct.Agg, ptr %[[A_ALLOCA]], i32 0, i32 0
// LLVM:   store i32 1, ptr %[[X]]
// LLVM:   %[[Y:.*]] = getelementptr inbounds nuw %struct.Agg, ptr %[[A_ALLOCA]], i32 0, i32 1
// LLVM:   store i32 2, ptr %[[Y]]
// LLVM:   br label %[[END:.*]]
// LLVM: [[END]]:
// LLVM:   ret void

// OGCG-LABEL: define{{.*}} void @_Z19test_agg_throw_trueb(
// OGCG:   %[[A:.*]] = alloca %struct.Agg
// OGCG:   %[[FLAG:.*]] = load i8, ptr %{{.*}}
// OGCG:   %[[BOOL:.*]] = icmp ne i8 %[[FLAG]], 0
// OGCG:   br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// OGCG:   store i32 0, ptr %[[EXC]]
// OGCG:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   %[[X:.*]] = getelementptr inbounds nuw %struct.Agg, ptr %[[A]], i32 0, i32 0
// OGCG:   store i32 1, ptr %[[X]]
// OGCG:   %[[Y:.*]] = getelementptr inbounds nuw %struct.Agg, ptr %[[A]], i32 0, i32 1
// OGCG:   store i32 2, ptr %[[Y]]
// OGCG:   br label %[[END:.*]]
// OGCG: [[END]]:
// OGCG:   ret void

void test_agg_throw_false(bool flag) {
  Agg a = flag ? Agg{1, 2} : throw 0;
}

// CIR-LABEL: cir.func{{.*}} @_Z20test_agg_throw_falseb(
// CIR:   %[[FLAG:.*]] = cir.alloca "flag"
// CIR:   %[[A:.*]] = cir.alloca "a"
// CIR:   %[[FLAG_VAL:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   cir.if %[[FLAG_VAL]] {
// CIR:     %[[X:.*]] = cir.get_member %[[A]][0] {name = "x"} : !cir.ptr<!rec_Agg> -> !cir.ptr<!s32i>
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     cir.store{{.*}} %[[ONE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:     %[[Y:.*]] = cir.get_member %[[A]][1] {name = "y"} : !cir.ptr<!rec_Agg> -> !cir.ptr<!s32i>
// CIR:     %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:     cir.store{{.*}} %[[TWO]], %[[Y]] : !s32i, !cir.ptr<!s32i>
// CIR:   } else {
// CIR:     %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:     cir.store{{.*}} %[[ZERO]], %[[EXC]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   }
// CIR:   cir.return

// LLVM-LABEL: define{{.*}} void @_Z20test_agg_throw_falseb(
// LLVM:   %[[FLAG_ALLOCA:.*]] = alloca i8
// LLVM:   %[[A_ALLOCA:.*]] = alloca %struct.Agg
// LLVM:   %[[BOOL:.*]] = trunc i8 %{{.*}} to i1
// LLVM:   br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[X:.*]] = getelementptr inbounds nuw %struct.Agg, ptr %[[A_ALLOCA]], i32 0, i32 0
// LLVM:   store i32 1, ptr %[[X]]
// LLVM:   %[[Y:.*]] = getelementptr inbounds nuw %struct.Agg, ptr %[[A_ALLOCA]], i32 0, i32 1
// LLVM:   store i32 2, ptr %[[Y]]
// LLVM:   br label %[[END:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// LLVM:   store i32 0, ptr %[[EXC]]
// LLVM:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[END]]:
// LLVM:   ret void

// OGCG-LABEL: define{{.*}} void @_Z20test_agg_throw_falseb(
// OGCG:   %[[A:.*]] = alloca %struct.Agg
// OGCG:   %[[FLAG:.*]] = load i8, ptr %{{.*}}
// OGCG:   %[[BOOL:.*]] = icmp ne i8 %[[FLAG]], 0
// OGCG:   br i1 %[[BOOL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   %[[X:.*]] = getelementptr inbounds nuw %struct.Agg, ptr %[[A]], i32 0, i32 0
// OGCG:   store i32 1, ptr %[[X]]
// OGCG:   %[[Y:.*]] = getelementptr inbounds nuw %struct.Agg, ptr %[[A]], i32 0, i32 1
// OGCG:   store i32 2, ptr %[[Y]]
// OGCG:   br label %[[END:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   %[[EXC:.*]] = call{{.*}} ptr @__cxa_allocate_exception
// OGCG:   store i32 0, ptr %[[EXC]]
// OGCG:   call void @__cxa_throw(ptr %[[EXC]], ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[END]]:
// OGCG:   ret void

int test_scalar_throw_true(bool flag, int x) {
  return flag ? throw 0 : x;
}

// CIR-LABEL: cir.func {{.*}} @_Z22test_scalar_throw_truebi(
// CIR:   %[[COND:.*]] = cir.load{{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   %{{.*}} = cir.ternary(%[[COND]], true {
// CIR:     %[[EXC:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   }, false {
// CIR:     %[[X:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:     cir.yield %[[X]] : !s32i
// CIR:   }) : (!cir.bool) -> !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z22test_scalar_throw_truebi(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   %{{.*}} = load i32
// LLVM:   ret i32

// OGCG-LABEL: define{{.*}} i32 @_Z22test_scalar_throw_truebi(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   ret i32

int test_scalar_throw_false(bool flag, int x) {
  return flag ? x : throw 0;
}

// CIR-LABEL: cir.func {{.*}} @_Z23test_scalar_throw_falsebi(
// CIR:   %{{.*}} = cir.ternary(%{{.*}}, true {
// CIR:     %[[X:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:     cir.yield %[[X]] : !s32i
// CIR:   }, false {
// CIR:     %[[EXC:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   }) : (!cir.bool) -> !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z23test_scalar_throw_falsebi(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable

// OGCG-LABEL: define{{.*}} i32 @_Z23test_scalar_throw_falsebi(
// OGCG:   br i1 %{{.*}}, label %{{.*}}, label %[[FALSE_BB:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable

void test_both_throw(bool flag) {
  flag ? throw 1 : throw 2;
}

// CIR-LABEL: cir.func {{.*}} @_Z15test_both_throwb(
// CIR:   cir.ternary(%{{.*}}, true {
// CIR:     cir.throw %{{.*}} : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   }, false {
// CIR:     cir.throw %{{.*}} : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   })

// LLVM-LABEL: define{{.*}} void @_Z15test_both_throwb(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable

// OGCG-LABEL: define{{.*}} void @_Z15test_both_throwb(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable

// The surviving arm of a glvalue conditional may compute its pointer inside
// the ternary region (e.g. loading a reference parameter); the result must be
// addressed through the cir.ternary result, which the region's yield carries
// out, not through the region-local pointer.
int &test_ref_cond_throw(bool c, int &x) {
  return c ? x : throw 0;
}

// CIR-LABEL: cir.func{{.*}} @_Z19test_ref_cond_throwbRi(
// CIR:   %[[C:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.bool>
// CIR:   %[[X_REF:.*]] = cir.alloca "x" {{.*}} init const : !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[C_VAL:.*]] = cir.load{{.*}} %[[C]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   %[[RES:.*]] = cir.ternary(%[[C_VAL]], true {
// CIR:     %[[X_PTR:.*]] = cir.load %[[X_REF]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:     cir.yield %[[X_PTR]] : !cir.ptr<!s32i>
// CIR-NEXT:   }, false {
// CIR:     cir.throw {{.*}} @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR:   cir.store %[[RES]], %[[RET_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM-LABEL: define{{.*}} ptr @_Z19test_ref_cond_throwbRi(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[X_PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   br label %[[PHI_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[PHI_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[X_PTR]], %[[TRUE_BB]] ]
// LLVM:   store ptr %[[PHI]], ptr %[[RET_ALLOCA:.*]], align 8
// LLVM:   %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM:   ret ptr %[[RET]]

// OGCG-LABEL: define{{.*}} ptr @_Z19test_ref_cond_throwbRi(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   %[[X_PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   br label %[[END:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[END]]:
// OGCG:   ret ptr %[[X_PTR]]

// Same shape with the conditional as the target of an assignment.
void test_assign_through_cond(bool c, int &x) {
  (c ? x : throw 0) = 5;
}

// CIR-LABEL: cir.func{{.*}} @_Z24test_assign_through_condbRi(
// CIR:   %[[X_REF:.*]] = cir.alloca "x" {{.*}} init const : !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR:   %[[RES:.*]] = cir.ternary(%{{.*}}, true {
// CIR:     %[[X_PTR:.*]] = cir.load %[[X_REF]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:     cir.yield %[[X_PTR]] : !cir.ptr<!s32i>
// CIR-NEXT:   }, false {
// CIR:     cir.throw {{.*}} @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %[[FIVE]], %[[RES]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} void @_Z24test_assign_through_condbRi(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[X_PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   br label %[[PHI_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[PHI_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[X_PTR]], %[[TRUE_BB]] ]
// LLVM:   store i32 5, ptr %[[PHI]], align 4
// LLVM:   ret void

// OGCG-LABEL: define{{.*}} void @_Z24test_assign_through_condbRi(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   %[[X_PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   br label %[[END:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[END]]:
// OGCG:   store i32 5, ptr %[[X_PTR]], align 4
// OGCG:   ret void

// Same shape where the surviving arm addresses a member through a pointer.
int &test_member_cond_throw(bool c, struct s6 *p) {
  return c ? p->f0 : throw 0;
}

// CIR-LABEL: cir.func{{.*}} @_Z22test_member_cond_throwbP2s6(
// CIR:   %[[P_ADDR:.*]] = cir.alloca "p" {{.*}} : !cir.ptr<!cir.ptr<!rec_s6>>
// CIR:   %[[RES:.*]] = cir.ternary(%{{.*}}, true {
// CIR:     %[[P:.*]] = cir.load{{.*}} %[[P_ADDR]] : !cir.ptr<!cir.ptr<!rec_s6>>, !cir.ptr<!rec_s6>
// CIR:     %[[F0:.*]] = cir.get_member %[[P]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>
// CIR:     cir.yield %[[F0]] : !cir.ptr<!s32i>
// CIR-NEXT:   }, false {
// CIR:     cir.throw {{.*}} @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   }) : (!cir.bool) -> !cir.ptr<!s32i>
// CIR:   cir.store %[[RES]], %{{.*}} : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM-LABEL: define{{.*}} ptr @_Z22test_member_cond_throwbP2s6(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[F0_PTR:.*]] = getelementptr{{.*}}%struct.s6, ptr %{{.*}}
// LLVM:   br label %[[PHI_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[PHI_BB]]:
// LLVM:   %[[PHI:.*]] = phi ptr [ %[[F0_PTR]], %[[TRUE_BB]] ]
// LLVM:   store ptr %[[PHI]], ptr %{{.*}}, align 8

// OGCG-LABEL: define{{.*}} ptr @_Z22test_member_cond_throwbP2s6(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   %[[F0_PTR:.*]] = getelementptr{{.*}}%struct.s6, ptr %{{.*}}
// OGCG:   br label %[[END:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[END]]:
// OGCG:   ret ptr %[[F0_PTR]]

// Aggregate conditional operators are emitted as cir.if; the branch regions
// are terminated after creation, so a throw arm must end at cir.unreachable
// with no trailing dead block (checked with CIR-NEXT below).

void take(Agg a);

// Baseline: both arms emit into the destination slot and the regions are
// closed with implicit terminators.
void test_agg_init_normal(bool c) {
  Agg a = c ? Agg{1, 2} : Agg{3, 4};
}

// CIR-LABEL: cir.func{{.*}} @_Z20test_agg_init_normalb(
// CIR:   %[[C:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.bool>
// CIR:   %[[A:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!rec_Agg>
// CIR:   %[[C_VAL:.*]] = cir.load{{.*}} %[[C]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   cir.if %[[C_VAL]] {
// CIR:     %[[X:.*]] = cir.get_member %[[A]][0] {name = "x"}
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     cir.store{{.*}} %[[ONE]], %[[X]]
// CIR:     %[[Y:.*]] = cir.get_member %[[A]][1] {name = "y"}
// CIR:     %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:     cir.store{{.*}} %[[TWO]], %[[Y]]
// CIR-NEXT:   } else {
// CIR:     %[[X2:.*]] = cir.get_member %[[A]][0] {name = "x"}
// CIR:     %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:     cir.store{{.*}} %[[THREE]], %[[X2]]
// CIR:     %[[Y2:.*]] = cir.get_member %[[A]][1] {name = "y"}
// CIR:     %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR:     cir.store{{.*}} %[[FOUR]], %[[Y2]]
// CIR-NEXT:   }
// CIR:   cir.return

// LLVM-LABEL: define{{.*}} void @_Z20test_agg_init_normalb(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   store i32 1, ptr %{{.*}}
// LLVM:   store i32 2, ptr %{{.*}}
// LLVM:   br label %[[END:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   store i32 3, ptr %{{.*}}
// LLVM:   store i32 4, ptr %{{.*}}
// LLVM:   br label %[[END]]
// LLVM: [[END]]:
// LLVM:   ret void

// OGCG-LABEL: define{{.*}} void @_Z20test_agg_init_normalb(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   store i32 1, ptr %{{.*}}
// OGCG:   store i32 2, ptr %{{.*}}
// OGCG:   br label %[[END:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   store i32 3, ptr %{{.*}}
// OGCG:   store i32 4, ptr %{{.*}}
// OGCG:   br label %[[END]]
// OGCG: [[END]]:
// OGCG:   ret void

// Assignment context: the conditional materializes into a temporary that is
// then assigned to the target.
void test_agg_assign_throw(bool c, Agg &a) {
  a = c ? throw 0 : Agg{1, 2};
}

// CIR-LABEL: cir.func{{.*}} @_Z21test_agg_assign_throwbR3Agg(
// CIR:   %[[C:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.bool>
// CIR:   %[[A_REF:.*]] = cir.alloca "a" {{.*}} init const : !cir.ptr<!cir.ptr<!rec_Agg>>
// CIR:   %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_Agg>
// CIR:   %[[C_VAL:.*]] = cir.load{{.*}} %[[C]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   cir.if %[[C_VAL]] {
// CIR:     %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   } else {
// CIR:     cir.get_member %[[TMP]][0] {name = "x"}
// CIR:     cir.get_member %[[TMP]][1] {name = "y"}
// CIR:   }
// CIR:   %[[A_VAL:.*]] = cir.load %[[A_REF]]
// CIR:   cir.call @_ZN3AggaSEOS_(%[[A_VAL]], %[[TMP]])

// LLVM-LABEL: define{{.*}} void @_Z21test_agg_assign_throwbR3Agg(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   call{{.*}} ptr @__cxa_allocate_exception
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   store i32 1, ptr %{{.*}}
// LLVM:   store i32 2, ptr %{{.*}}
// LLVM:   br label %[[END:.*]]
// LLVM: [[END]]:
// LLVM:   call{{.*}} ptr @_ZN3AggaSEOS_(

// OGCG-LABEL: define{{.*}} void @_Z21test_agg_assign_throwbR3Agg(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   call{{.*}} ptr @__cxa_allocate_exception
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   store i32 1, ptr %{{.*}}
// OGCG:   store i32 2, ptr %{{.*}}
// OGCG:   br label %[[END:.*]]
// OGCG: [[END]]:
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 8

// Nested conditional: the inner throw arm terminates the inner cir.if region
// directly.
void test_agg_nested_throw(bool c1, bool c2) {
  Agg a = c1 ? (c2 ? throw 0 : Agg{1, 2}) : Agg{3, 4};
}

// CIR-LABEL: cir.func{{.*}} @_Z21test_agg_nested_throwbb(
// CIR:   %[[A:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!rec_Agg>
// CIR:   cir.if %{{.*}} {
// CIR:     %[[C2_VAL:.*]] = cir.load{{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[C2_VAL]] {
// CIR:       %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:       cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:       cir.unreachable
// CIR-NEXT:     } else {
// CIR:       cir.get_member %[[A]][0] {name = "x"}
// CIR:       cir.get_member %[[A]][1] {name = "y"}
// CIR:     }
// CIR:   } else {
// CIR:     cir.get_member %[[A]][0] {name = "x"}
// CIR:     cir.get_member %[[A]][1] {name = "y"}
// CIR:   }
// CIR:   cir.return

// LLVM-LABEL: define{{.*}} void @_Z21test_agg_nested_throwbb(
// LLVM:   br i1 %{{.*}}, label %[[OUTER_TRUE:.*]], label %[[OUTER_FALSE:.*]]
// LLVM: [[OUTER_TRUE]]:
// LLVM:   br i1 %{{.*}}, label %[[INNER_TRUE:.*]], label %[[INNER_FALSE:.*]]
// LLVM: [[INNER_TRUE]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[INNER_FALSE]]:
// LLVM:   store i32 1, ptr %{{.*}}
// LLVM:   store i32 2, ptr %{{.*}}
// LLVM: [[OUTER_FALSE]]:
// LLVM:   store i32 3, ptr %{{.*}}
// LLVM:   store i32 4, ptr %{{.*}}

// OGCG-LABEL: define{{.*}} void @_Z21test_agg_nested_throwbb(
// OGCG:   br i1 %{{.*}}, label %[[OUTER_TRUE:.*]], label %[[OUTER_FALSE:.*]]
// OGCG: [[OUTER_TRUE]]:
// OGCG:   br i1 %{{.*}}, label %[[INNER_TRUE:.*]], label %[[INNER_FALSE:.*]]
// OGCG: [[INNER_TRUE]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[INNER_FALSE]]:
// OGCG:   store i32 1, ptr %{{.*}}
// OGCG:   store i32 2, ptr %{{.*}}
// OGCG: [[OUTER_FALSE]]:
// OGCG:   store i32 3, ptr %{{.*}}
// OGCG:   store i32 4, ptr %{{.*}}

// Call-argument context: the conditional materializes the argument temporary.
void test_agg_arg_throw(bool c) {
  take(c ? throw 0 : Agg{1, 2});
}

// CIR-LABEL: cir.func{{.*}} @_Z18test_agg_arg_throwb(
// CIR:   %[[TMP:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_Agg>
// CIR:   cir.if %{{.*}} {
// CIR:     %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   } else {
// CIR:     cir.get_member %[[TMP]][0] {name = "x"}
// CIR:     cir.get_member %[[TMP]][1] {name = "y"}
// CIR:   }
// CIR:   %[[ARG:.*]] = cir.load{{.*}} %[[TMP]] : !cir.ptr<!rec_Agg>, !rec_Agg
// CIR:   cir.call @_Z4take3Agg(%[[ARG]])

// LLVM-LABEL: define{{.*}} void @_Z18test_agg_arg_throwb(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   store i32 1, ptr %{{.*}}
// LLVM:   store i32 2, ptr %{{.*}}
// LLVM:   br label %[[END:.*]]
// LLVM: [[END]]:
// LLVM:   call void @_Z4take3Agg(

// OGCG-LABEL: define{{.*}} void @_Z18test_agg_arg_throwb(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   store i32 1, ptr %{{.*}}
// OGCG:   store i32 2, ptr %{{.*}}
// OGCG:   br label %[[END:.*]]
// OGCG: [[END]]:
// OGCG:   call void @_Z4take3Agg(
