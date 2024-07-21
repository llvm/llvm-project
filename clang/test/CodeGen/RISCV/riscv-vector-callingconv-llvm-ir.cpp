// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -std=c++11 -triple riscv64 -target-feature +v \
// RUN:   -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-LLVM %s
// RUN: %clang_cc1 -std=c++11 -triple riscv64 -mriscv-abi-vlen=256 -target-feature +v \
// RUN:   -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-LLVM-ABI-VLEN %s

#include <riscv_vector.h>

// CHECK-LLVM: call riscv_vector_cc <vscale x 2 x i32> @_Z3baru15__rvv_int32m1_t
vint32m1_t __attribute__((riscv_vector_cc)) bar(vint32m1_t input);
vint32m1_t test_vector_cc_attr(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = bar(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}

// CHECK-LLVM: call riscv_vector_cc <vscale x 2 x i32> @_Z3baru15__rvv_int32m1_t
[[riscv::vector_cc]] vint32m1_t bar(vint32m1_t input);
vint32m1_t test_vector_cc_attr2(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = bar(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}

// CHECK-LLVM: call <vscale x 2 x i32> @_Z3bazu15__rvv_int32m1_t
vint32m1_t baz(vint32m1_t input);
vint32m1_t test_no_vector_cc_attr(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = baz(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}

// CHECK-LLVM: define dso_local void @_Z14test_vls_no_ccDv4_i(i128 noundef %arg.coerce)
// CHECK-LLVM-ABI-VLEN: define dso_local void @_Z14test_vls_no_ccDv4_i(<vscale x 1 x i32> noundef %arg.coerce)
void test_vls_no_cc(__attribute__((vector_size(16))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc void @_Z25test_vls_default_abi_vlenDv4_i(<vscale x 2 x i32> noundef %arg.coerce)
// CHECK-LLVM-ABI-VLEN: define dso_local riscv_vls_cc void @_Z25test_vls_default_abi_vlenDv4_i(<vscale x 2 x i32> noundef %arg.coerce)
[[riscv::vls_cc]] void test_vls_default_abi_vlen(__attribute__((vector_size(16))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc void @_Z45test_vls_default_abi_vlen_unsupported_featureDv8_DF16_(<vscale x 2 x i32> noundef %arg.coerce)
// CHECK-LLVM-ABI-VLEN: define dso_local riscv_vls_cc void @_Z45test_vls_default_abi_vlen_unsupported_featureDv8_DF16_(<vscale x 2 x i32> noundef %arg.coerce)
[[riscv::vls_cc]] void test_vls_default_abi_vlen_unsupported_feature(__attribute__((vector_size(16))) _Float16 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc void @_Z21test_vls_256_abi_vlenDv4_i(<vscale x 1 x i32> noundef %arg.coerce)
// CHECK-LLVM-ABI-VLEN: define dso_local riscv_vls_cc void @_Z21test_vls_256_abi_vlenDv4_i(<vscale x 1 x i32> noundef %arg.coerce)
[[riscv::vls_cc(256)]] void test_vls_256_abi_vlen(__attribute__((vector_size(16))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc void @_Z22test_vls_least_elementDv2_i(<vscale x 1 x i32> noundef %arg.coerce)
// CHECK-LLVM-ABI-VLEN: define dso_local riscv_vls_cc void @_Z22test_vls_least_elementDv2_i(<vscale x 1 x i32> noundef %arg.coerce)
[[riscv::vls_cc(1024)]] void test_vls_least_element(__attribute__((vector_size(8))) int arg) {}
