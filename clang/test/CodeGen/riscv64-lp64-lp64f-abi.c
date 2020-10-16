// RUN: %clang_cc1 -triple riscv64 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-abi lp64f -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// This file contains test cases that will have the same output for the lp64
// and lp64f ABIs.

#include <stddef.h>
#include <stdint.h>

struct large {
  int64_t a, b, c, d;
};

typedef unsigned char v32i8 __attribute__((vector_size(32)));

// Scalars passed on the stack should not have signext/zeroext attributes
// (they are anyext).

// CHECK-LABEL: define signext i32 @f_scalar_stack_1(i32 signext %a, i128 %b, double %c, fp128 %d, <32 x i8>* %0, i8 zeroext %f, i8 %g, i8 %h)
int f_scalar_stack_1(int32_t a, __int128_t b, double c, long double d, v32i8 e,
                     uint8_t f, int8_t g, uint8_t h) {
  return g + h;
}

// Ensure that scalars passed on the stack are still determined correctly in
// the presence of large return values that consume a register due to the need
// to pass a pointer.

// CHECK-LABEL: define void @f_scalar_stack_2(%struct.large* noalias sret(%struct.large) align 8 %agg.result, double %a, i128 %b, fp128 %c, <32 x i8>* %0, i8 zeroext %e, i8 %f, i8 %g)
struct large f_scalar_stack_2(double a, __int128_t b, long double c, v32i8 d,
                              uint8_t e, int8_t f, uint8_t g) {
  return (struct large){a, e, f, g};
}
