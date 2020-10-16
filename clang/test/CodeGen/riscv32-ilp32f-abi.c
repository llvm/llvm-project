// RUN: %clang_cc1 -triple riscv32 -target-feature +f -target-abi ilp32f -emit-llvm %s -o - \
// RUN:     | FileCheck %s

#include <stdint.h>

// Doubles are still passed in GPRs, so the 'e' argument will be anyext as
// GPRs are exhausted.

// CHECK: define void @f_fpr_tracking(double %a, double %b, double %c, double %d, i8 %e)
void f_fpr_tracking(double a, double b, double c, double d, int8_t e) {}

// Lowering for doubles is unnmodified, as 64 > FLEN.

struct double_s { double d; };

// CHECK: define void @f_double_s_arg(i64 %a.coerce)
void f_double_s_arg(struct double_s a) {}

// CHECK: define i64 @f_ret_double_s()
struct double_s f_ret_double_s() {
  return (struct double_s){1.0};
}

struct double_double_s { double d; double e; };

// CHECK: define void @f_double_double_s_arg(%struct.double_double_s* %a)
void f_double_double_s_arg(struct double_double_s a) {}

// CHECK: define void @f_ret_double_double_s(%struct.double_double_s* noalias sret(%struct.double_double_s) align 8 %agg.result)
struct double_double_s f_ret_double_double_s() {
  return (struct double_double_s){1.0, 2.0};
}

struct double_int8_s { double d; int64_t i; };

struct int_double_s { int a; double b; };

// CHECK: define void @f_int_double_s_arg(%struct.int_double_s* %a)
void f_int_double_s_arg(struct int_double_s a) {}

// CHECK: define void @f_ret_int_double_s(%struct.int_double_s* noalias sret(%struct.int_double_s) align 8 %agg.result)
struct int_double_s f_ret_int_double_s() {
  return (struct int_double_s){1, 2.0};
}

