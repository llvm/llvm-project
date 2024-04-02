// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-LLVM %s
// RUN: %clang_cc1 -std=c23 -triple riscv64 -target-feature +v \
// RUN:   -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-LLVM %s

#include <riscv_vector.h>

// CHECK-LLVM: call riscv_vector_cc <vscale x 2 x i32> @bar
vint32m1_t __attribute__((riscv_vector_cc)) bar(vint32m1_t input);
vint32m1_t test_vector_cc_attr(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = bar(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}

// CHECK-LLVM: call riscv_vector_cc <vscale x 2 x i32> @bar
[[riscv::vector_cc]] vint32m1_t bar(vint32m1_t input);
vint32m1_t test_vector_cc_attr2(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = bar(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}

// CHECK-LLVM: call <vscale x 2 x i32> @baz
vint32m1_t baz(vint32m1_t input);
vint32m1_t test_no_vector_cc_attr(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = baz(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}
