// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -foverflow-behavior-types \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow -emit-llvm -o - -std=c++14 | FileCheck %s

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __nowrap __attribute__((overflow_behavior(no_wrap)))

typedef int __wrap wrap_int;
typedef char __wrap wrap_char;
typedef int __nowrap nowrap_int;
typedef unsigned int __wrap u_wrap_int;
typedef unsigned int __nowrap u_nowrap_int;

// CHECK-LABEL: define {{.*}} @_Z30conditional_operator_promotionbU11ObtWrap_cU13ObtNoWrap_ii
void conditional_operator_promotion(bool cond, wrap_char w, nowrap_int nw, int i) {
  // OBT wins over regular integer.
  // CHECK: cond.end:
  // CHECK-NEXT: %cond1 = phi i8
  // CHECK-NEXT: store i8 %cond1, ptr %r1
  // CHECK-NEXT: %{{.*}} = load i8, ptr %r1
  // CHECK-NEXT: add i8
  auto r1 = cond ? w : i;
  (void)(r1 + 2147483647);

  // nowrap wins over wrap.
  // CHECK: cond.end6:
  // CHECK-NEXT: %cond7 = phi i32
  // CHECK-NEXT: store i32 %cond7, ptr %r2
  // CHECK-NEXT: %{{.*}} = load i32, ptr %r2
  // CHECK-NEXT: call { i32, i1 } @llvm.sadd.with.overflow.i32
  auto r2 = cond ? w : nw;
  (void)(r2 + 2147483647);
}

// CHECK-LABEL: define {{.*}} @_Z20promotion_rules_testU11ObtWrap_iU13ObtNoWrap_iU11ObtWrap_jU13ObtNoWrap_j
void promotion_rules_test(wrap_int sw, nowrap_int snw, u_wrap_int uw, u_nowrap_int unw) {
  // Unsigned is favored over signed for same-behavior OBTs.
  // CHECK: add i32
  auto r1 = sw + uw;
  (void)r1;

  // nowrap is favored over wrap. Result is unsigned nowrap.
  // CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32
  auto r2 = sw + unw;
  (void)r2;

  // nowrap is favored over wrap. Result is signed nowrap.
  // CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32
  auto r3 = uw + snw;
  (void)r3;
}
