// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

int f1() {
  int i;
  return i;
}

// CHECK: module
// CHECK: cir.func @f1() -> !s32i
// CHECK:    %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:    %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"] {alignment = 4 : i64}
// CHECK:    %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.store %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.return %[[R]] : !s32i

int f2() {
  const int i = 2;
  return i;
}

// CHECK: cir.func @f2() -> !s32i
// CHECK:    %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:    %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init, const] {alignment = 4 : i64}
// CHECK:    %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CHECK:    cir.store %[[TWO]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.store %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.return %[[R]] : !s32i

int f3(int i) {
  return i;
}

// CHECK: cir.func @f3(%[[ARG:.*]]: !s32i loc({{.*}})) -> !s32i
// CHECK:   %[[ARG_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   cir.store %[[ARG]], %[[ARG_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[ARG_VAL:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store %[[ARG_VAL]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.return %[[R]] : !s32i

int f4(const int i) {
  return i;
}

// CHECK: cir.func @f4(%[[ARG:.*]]: !s32i loc({{.*}})) -> !s32i
// CHECK:   %[[ARG_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init, const] {alignment = 4 : i64}
// CHECK:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   cir.store %[[ARG]], %[[ARG_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[ARG_VAL:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store %[[ARG_VAL]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.return %[[R]] : !s32i

int *f5() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 43;
  return p;
}

// CHECK:      cir.func @f5() -> !cir.ptr<!s32i>
// CHECK-NEXT:   %[[RET_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CHECK-NEXT:   %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init] {alignment = 8 : i64}
// CHECK-NEXT:   %[[NULLPTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store %[[NULLPTR]], %[[P_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %[[X_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK-NEXT:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:     cir.store %[[ZERO]], %[[X_ADDR]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:     cir.store %[[X_ADDR]], %[[P_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT:     %[[FOURTYTWO:.*]] = cir.const #cir.int<42> : !s32i
// CHECK-NEXT:     %[[P:.*]] = cir.load deref %[[P_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:     cir.store %[[FOURTYTWO]], %[[P]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[FOURTYTHREE:.*]] = cir.const #cir.int<43> : !s32i
// CHECK-NEXT:   %[[P:.*]] = cir.load deref %[[P_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store %[[FOURTYTHREE]], %[[P]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %[[P:.*]] = cir.load %[[P_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store %[[P]], %[[RET_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT:   %[[RET_VAL:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.return %[[RET_VAL]] : !cir.ptr<!s32i>
