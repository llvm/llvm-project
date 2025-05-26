// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

int f1() {
  int i;
  return i;
}

// CHECK: module
// CHECK: cir.func @_Z2f1v() -> !s32i
// CHECK:    %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:    %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"] {alignment = 4 : i64}
// CHECK:    %[[I:.*]] = cir.load{{.*}} %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.store{{.*}} %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[R:.*]] = cir.load{{.*}} %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.return %[[R]] : !s32i

int f2() {
  const int i = 2;
  return i;
}

// CHECK: cir.func @_Z2f2v() -> !s32i
// CHECK:    %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:    %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init, const] {alignment = 4 : i64}
// CHECK:    %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CHECK:    cir.store{{.*}} %[[TWO]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[I:.*]] = cir.load{{.*}} %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.store{{.*}} %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[R:.*]] = cir.load{{.*}} %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.return %[[R]] : !s32i

int f3(int i) {
  return i;
}

// CHECK: cir.func @_Z2f3i(%[[ARG:.*]]: !s32i loc({{.*}})) -> !s32i
// CHECK:   %[[ARG_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   cir.store{{.*}} %[[ARG]], %[[ARG_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[ARG_VAL:.*]] = cir.load{{.*}} %[[ARG_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store{{.*}} %[[ARG_VAL]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[R:.*]] = cir.load{{.*}} %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.return %[[R]] : !s32i

int f4(const int i) {
  return i;
}

// CHECK: cir.func @_Z2f4i(%[[ARG:.*]]: !s32i loc({{.*}})) -> !s32i
// CHECK:   %[[ARG_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init, const] {alignment = 4 : i64}
// CHECK:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   cir.store{{.*}} %[[ARG]], %[[ARG_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[ARG_VAL:.*]] = cir.load{{.*}} %[[ARG_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store{{.*}} %[[ARG_VAL]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[R:.*]] = cir.load{{.*}} %[[RV]] : !cir.ptr<!s32i>, !s32i
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

// CHECK:      cir.func @_Z2f5v() -> !cir.ptr<!s32i>
// CHECK-NEXT:   %[[RET_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CHECK-NEXT:   %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init] {alignment = 8 : i64}
// CHECK-NEXT:   %[[NULLPTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store{{.*}} %[[NULLPTR]], %[[P_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %[[X_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK-NEXT:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:     cir.store{{.*}} %[[ZERO]], %[[X_ADDR]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:     cir.store{{.*}} %[[X_ADDR]], %[[P_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT:     %[[FOURTYTWO:.*]] = cir.const #cir.int<42> : !s32i
// CHECK-NEXT:     %[[P:.*]] = cir.load deref{{.*}} %[[P_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:     cir.store{{.*}} %[[FOURTYTWO]], %[[P]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[FOURTYTHREE:.*]] = cir.const #cir.int<43> : !s32i
// CHECK-NEXT:   %[[P:.*]] = cir.load deref{{.*}} %[[P_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store{{.*}} %[[FOURTYTHREE]], %[[P]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %[[P:.*]] = cir.load{{.*}} %[[P_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store{{.*}} %[[P]], %[[RET_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT:   %[[RET_VAL:.*]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.return %[[RET_VAL]] : !cir.ptr<!s32i>

using size_type = unsigned long;
using _Tp = unsigned long;

size_type max_size() {
  return size_type(~0) / sizeof(_Tp);
}

// CHECK: cir.func @_Z8max_sizev() -> !u64i
// CHECK:   %0 = cir.alloca !u64i, !cir.ptr<!u64i>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.int<0> : !s32i
// CHECK:   %2 = cir.unary(not, %1) : !s32i, !s32i
// CHECK:   %3 = cir.cast(integral, %2 : !s32i), !u64i
// CHECK:   %4 = cir.const #cir.int<8> : !u64i
// CHECK:   %5 = cir.binop(div, %3, %4) : !u64i
// CHECK:   cir.store{{.*}} %5, %0 : !u64i, !cir.ptr<!u64i>
// CHECK:   %6 = cir.load{{.*}} %0 : !cir.ptr<!u64i>, !u64i
// CHECK:   cir.return %6 : !u64i
// CHECK:   }

void ref_arg(int &x) {
  int y = x;
  x = 3;
}

// CHECK: cir.func @_Z7ref_argRi(%[[ARG:.*]]: !cir.ptr<!s32i> {{.*}})
// CHECK:   %[[X_REF_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["x", init, const] {alignment = 8 : i64}
// CHECK:   %[[Y_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init] {alignment = 4 : i64}
// CHECK:   cir.store{{.*}} %[[ARG]], %[[X_REF_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK:   %[[X_REF:.*]] = cir.load{{.*}} %[[X_REF_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:   %[[Y:.*]] = cir.load{{.*}} %[[X_REF]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store{{.*}} %[[Y]], %[[Y_ADDR]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CHECK:   %[[X_REF:.*]] = cir.load{{.*}} %[[X_REF_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:   cir.store{{.*}} %[[THREE]], %[[X_REF]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.return

short gs;
short &return_ref() {
  return gs;
}

// CHECK: cir.func @_Z10return_refv() -> !cir.ptr<!s16i>
// CHECK:   %[[RETVAL_ADDR:.*]] = cir.alloca !cir.ptr<!s16i>, !cir.ptr<!cir.ptr<!s16i>>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %[[GS_ADDR:.*]] = cir.get_global @gs : !cir.ptr<!s16i>
// CHECK:   cir.store{{.*}} %[[GS_ADDR]], %[[RETVAL_ADDR]] : !cir.ptr<!s16i>, !cir.ptr<!cir.ptr<!s16i>>
// CHECK:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]] : !cir.ptr<!cir.ptr<!s16i>>, !cir.ptr<!s16i>
// CHECK:   cir.return %[[RETVAL]] : !cir.ptr<!s16i>

void ref_local(short x) {
  short &y = x;
}

// CHECK: cir.func @_Z9ref_locals(%[[ARG:.*]]: !s16i {{.*}})
// CHECK:   %[[X_ADDR:.*]] = cir.alloca !s16i, !cir.ptr<!s16i>, ["x", init] {alignment = 2 : i64}
// CHECK:   %[[Y_REF_ADDR:.*]] = cir.alloca !cir.ptr<!s16i>, !cir.ptr<!cir.ptr<!s16i>>, ["y", init, const] {alignment = 8 : i64}
// CHECK:   cir.store{{.*}} %[[ARG]], %[[X_ADDR]] : !s16i, !cir.ptr<!s16i>
// CHECK:   cir.store{{.*}} %[[X_ADDR]], %[[Y_REF_ADDR]] : !cir.ptr<!s16i>, !cir.ptr<!cir.ptr<!s16i>>

enum A {
  A_one,
  A_two
};
enum A a;

// CHECK:   cir.global external @a = #cir.int<0> : !u32i

enum B : int;
enum B b;

// CHECK:   cir.global external @b = #cir.int<0> : !s32i

enum C : int {
  C_one,
  C_two
};
enum C c;

// CHECK:   cir.global external @c = #cir.int<0> : !s32i

enum class D : int;
enum D d;

// CHECK:   cir.global external @d = #cir.int<0> : !s32i
