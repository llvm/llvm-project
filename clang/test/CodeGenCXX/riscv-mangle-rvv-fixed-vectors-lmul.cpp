// Verify that the LMUL suffix in the mangled name of a riscv_rvv_vector_bits
// typedef comes from the type the typedef was written against, not from VLEN.
// Two typedefs holding the same number of lanes of the same element type but
// derived from different LMULs must mangle
// differently, and - because the suffix no longer depends on VLEN - the same
// source mangles identically at zvl128b and zvl256b.

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl128b \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl256b \
// RUN:   | FileCheck %s

// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

// All three hold 2 i32 lanes but come from different LMULs.
typedef vint32mf2_t mf2_64 __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m1_t  m1_64  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m2_t  m2_64  __attribute__((riscv_rvv_vector_bits(64)));

// Overloading on the three types only compiles if they are distinct types.
void f(mf2_64);
void f(m1_64);
void f(m2_64);

void use(mf2_64 a, m1_64 b, m2_64 c) {
  f(a);
  f(b);
  f(c);
}

// CHECK: declare {{.*}}@_Z1f9__RVV_VLSIu16__rvv_int32mf2_tLj64EE
// CHECK: declare {{.*}}@_Z1f9__RVV_VLSIu15__rvv_int32m1_tLj64EE
// CHECK: declare {{.*}}@_Z1f9__RVV_VLSIu15__rvv_int32m2_tLj64EE
