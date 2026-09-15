// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -fexperimental-overflow-behavior-types \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow -emit-llvm -o - -std=c++14 | FileCheck %s

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_trap __attribute__((overflow_behavior(trap)))

typedef int __ob_wrap wrap_int;
typedef char __ob_wrap wrap_char;
typedef int __ob_trap no_trap_int;
typedef unsigned int __ob_wrap u_wrap_int;
typedef unsigned int __ob_trap u_no_trap_int;

// CHECK-LABEL: define {{.*}} @_Z30conditional_operator_promotionbU8ObtWrap_cU8ObtTrap_ii
void conditional_operator_promotion(bool cond, wrap_char w, no_trap_int nw, int i) {
  // CHECK: cond.end:
  // CHECK-NEXT: %cond1 = phi i32
  // CHECK-NEXT: store i32 %cond1, ptr %r1
  // CHECK-NEXT: %{{.*}} = load i32, ptr %r1
  // CHECK-NEXT: add i32
  auto r1 = cond ? w : i;
  (void)(r1 + 2147483647);

  // no_trap wins over wrap.
  // CHECK: cond.end6:
  // CHECK-NEXT: %cond7 = phi i32
  // CHECK-NEXT: store i32 %cond7, ptr %r2
  // CHECK-NEXT: %{{.*}} = load i32, ptr %r2
  // CHECK-NEXT: call { i32, i1 } @llvm.sadd.with.overflow.i32
  auto r2 = cond ? w : nw;
  (void)(r2 + 2147483647);
}

// CHECK-LABEL: define {{.*}} @_Z20promotion_rules_testU8ObtWrap_iU8ObtTrap_iU8ObtWrap_jU8ObtTrap_j
void promotion_rules_test(wrap_int sw, no_trap_int snw, u_wrap_int uw, u_no_trap_int unw) {
  // Unsigned is favored over signed for same-behavior OBTs.
  // CHECK: add i32
  auto r1 = sw + uw;
  (void)r1;

  // trap is favored over wrap. Result is unsigned no_trap.
  // CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32
  auto r2 = sw + unw;
  (void)r2;

  // trap is favored over wrap. Result is unsigned trap (unsigned int + int → unsigned int in C).
  // CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32
  auto r3 = uw + snw;
  (void)r3;
}
