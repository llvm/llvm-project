// REQUIRES: riscv-registered-target
// RUN: not %clang_cc1 -triple riscv64 -target-feature +zifencei -target-feature +m -target-feature +a \
// RUN:  -emit-llvm-only %s 2>&1 | FileCheck %s

#include <riscv_vector.h>

void test_builtin() {
// CHECK: error: '__builtin_rvv_vsetvli' needs target feature zve32x
  __riscv_vsetvl_e8m8(1);
}

void test_rvv_i32_type() {
// CHECK: error: RISC-V type 'vint32m1_t' (aka '__rvv_int32m1_t') requires the 'zve32x' extension
  vint32m1_t v;
}

void test_rvv_f32_type() {
// CHECK: error: RISC-V type 'vfloat32m1_t' (aka '__rvv_float32m1_t') requires the 'zve32f' extension
  vfloat32m1_t v;
}

void test_rvv_f64_type() {
// CHECK: error: RISC-V type 'vfloat64m1_t' (aka '__rvv_float64m1_t') requires the 'zve64d' extension
  vfloat64m1_t v;
}

// CHECK: error: duplicate 'arch=' in the 'target' attribute string;
__attribute__((target("arch=rv64gc;arch=rv64gc_zbb"))) void testMultiArchSelectLast() {}
// CHECK: error: duplicate 'cpu=' in the 'target' attribute string;
__attribute__((target("cpu=sifive-u74;cpu=sifive-u54"))) void testMultiCpuSelectLast() {}
// CHECK: error: duplicate 'tune=' in the 'target' attribute string;
__attribute__((target("tune=sifive-u74;tune=sifive-u54"))) void testMultiTuneSelectLast() {}
