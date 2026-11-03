// Verify that the LMUL suffix, or the mask ratio, in the mangled name of a
// riscv_rvv_vector_bits typedef comes from the type the typedef was written
// against, not from VLEN. Two typedefs holding the same number of lanes of the
// same element type but derived from different LMULs or ratios must mangle
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

// Masks: b8_8 and b4_8 both hold 8 lanes, b32_2 and b64_2 both hold 2 lanes
// in a single byte.
typedef vbool8_t  b8_8  __attribute__((riscv_rvv_vector_bits(8)));
typedef vbool4_t  b4_8  __attribute__((riscv_rvv_vector_bits(8)));
typedef vbool32_t b32_2 __attribute__((riscv_rvv_vector_bits(2)));
typedef vbool64_t b64_2 __attribute__((riscv_rvv_vector_bits(2)));

void g(b8_8);
void g(b4_8);
void g(b32_2);
void g(b64_2);

void use_b(b8_8 a, b4_8 b, b32_2 c, b64_2 d) {
  g(a);
  g(b);
  g(c);
  g(d);
}

// CHECK: declare {{.*}}@_Z1g9__RVV_VLSIu13__rvv_bool8_tLj8EE
// CHECK: declare {{.*}}@_Z1g9__RVV_VLSIu13__rvv_bool4_tLj8EE
// CHECK: declare {{.*}}@_Z1g9__RVV_VLSIu14__rvv_bool32_tLj2EE
// CHECK: declare {{.*}}@_Z1g9__RVV_VLSIu14__rvv_bool64_tLj2EE
