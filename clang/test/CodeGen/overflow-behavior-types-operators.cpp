// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -fexperimental-overflow-behavior-types \
// RUN: -fsanitize=signed-integer-overflow -emit-llvm -o - -std=c++14 | FileCheck %s

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_trap __attribute__((overflow_behavior(trap)))

typedef int __ob_wrap wrap_int;
typedef int __ob_trap no_trap_int;
typedef unsigned int __ob_wrap u_wrap_int;
typedef unsigned int __ob_trap u_no_trap_int;

//===----------------------------------------------------------------------===//
// Compound Assignment Operators
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}} @_Z28compound_assignment_operatorv
void compound_assignment_operator() {
  wrap_int a = 1;
  // CHECK: add i32
  a += 1;

  no_trap_int b = 1;
  // CHECK: llvm.sadd.with.overflow.i32
  b += 1;

  u_wrap_int c = 1;
  // CHECK: sub i32
  c -= 1;

  u_no_trap_int d = 1;
  // CHECK: llvm.usub.with.overflow.i32
  d -= 1;

  wrap_int e = 2;
  // CHECK: mul i32
  e *= 2;

  no_trap_int f = 2;
  // CHECK: llvm.smul.with.overflow.i32
  f *= 2;
}

//===----------------------------------------------------------------------===//
// Bitwise and Shift Operators
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}} @_Z27bitwise_and_shift_operatorsv
void bitwise_and_shift_operators() {
  wrap_int a = 1;
  // CHECK: shl i32
  // No overflow check for shifts
  a <<= 1;

  no_trap_int b = 1;
  // CHECK: ashr i32
  // No overflow check for shifts
  b >>= 1;

  wrap_int c = 1;
  // CHECK: and i32
  c &= 1;

  no_trap_int d = 1;
  // CHECK: xor i32
  d ^= 1;

  u_wrap_int e = 1;
  // CHECK: or i32
  e |= 1;
}
