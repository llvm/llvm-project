// Verify CodeGen coercion for shorter riscv_rvv_vector_bits typedefs.
//
// A shorter typedef keeps the LMUL of the sizeless type it was written
// against, so it is passed in the same register group as the full-size
// typedef; only the lanes it declares hold defined data.
//
// VLEN=128 (vscale=2): vint32m1_t has 4 i32 lanes at full size.
//   - bits(128) -> <4 x i32> -> <vscale x 2 x i32> (m1, all 4 lanes used)
//   - bits(64)  -> <2 x i32> -> <vscale x 2 x i32> (m1, upper 2 lanes garbage)
//   - bits(32)  -> <1 x i32> -> <vscale x 2 x i32> (m1, upper 3 lanes garbage)

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -mvscale-min=2 -mvscale-max=2 \
// RUN:   -O1 -emit-llvm -o - %s | FileCheck %s

// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

typedef vint32m1_t fixed_i32x4 __attribute__((riscv_rvv_vector_bits(128)));
typedef vint32m1_t fixed_i32x2 __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m1_t fixed_i32x1 __attribute__((riscv_rvv_vector_bits(32)));
typedef vint8m1_t  fixed_i8x4  __attribute__((riscv_rvv_vector_bits(32)));

// Full-size case still works the same as the strict path.
// CHECK-LABEL: define dso_local{{.*}} <vscale x 2 x i32> @id_i32x4(
// CHECK-SAME:    <vscale x 2 x i32> {{.*}}%x.coerce)
// CHECK:         ret <vscale x 2 x i32> %x.coerce
fixed_i32x4 id_i32x4(fixed_i32x4 x) { return x; }

// Half the lanes, same LMUL: still <vscale x 2 x i32>, not <vscale x 1 x i32>.
// CHECK-LABEL: define dso_local{{.*}} <vscale x 2 x i32> @id_i32x2(
// CHECK-SAME:    <vscale x 2 x i32> {{.*}}%x.coerce)
fixed_i32x2 id_i32x2(fixed_i32x2 x) { return x; }

// A single lane still occupies a whole LMUL=1 register group.
// CHECK-LABEL: define dso_local{{.*}} <vscale x 2 x i32> @id_i32x1(
// CHECK-SAME:    <vscale x 2 x i32> {{.*}}%x.coerce)
fixed_i32x1 id_i32x1(fixed_i32x1 x) { return x; }

// 4 i8 lanes = 32 bits out of a 128-bit LMUL=1 group, which holds 8 i8 lanes
// at vscale=1.
// CHECK-LABEL: define dso_local{{.*}} <vscale x 8 x i8> @id_i8x4(
// CHECK-SAME:    <vscale x 8 x i8> {{.*}}%x.coerce)
fixed_i8x4 id_i8x4(fixed_i8x4 x) { return x; }

// Note: implicit conversion between shorter fixed types and the sizeless
// VLA type (e.g. calling sizeless_callee(fixed_i32x2)) is handled by
// Phase 3 and is intentionally not exercised here.
