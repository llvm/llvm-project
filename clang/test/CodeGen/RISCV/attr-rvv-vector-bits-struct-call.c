// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v \
// RUN:   -mvscale-min=4 -mvscale-max=4 -emit-llvm -o - %s | FileCheck %s

#include <riscv_vector.h>

typedef vint32m1_t   fixed_int32m1_t   __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint32m4_t   fixed_int32m4_t   __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));
typedef vint32mf2_t  fixed_int32mf2_t  __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));
typedef vfloat64m1_t fixed_float64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
// Mask types. getRVVEltType returns UnsignedCharTy, so LLVM type is <N x i8>.
// vbool1_t  = nxv64i1;  VScale=4 -> ExpectedSize=256 bits -> Clang <32 x i8>
// vbool4_t  = nxv16i1;  VScale=4 -> ExpectedSize=64  bits -> Clang <8  x i8>
// vbool64_t = nxv1i1;   VScale=4 -> ExpectedSize=4   bits -> Clang <1  x i8> (sub-byte)
typedef vbool1_t  fixed_bool1_t  __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vbool4_t  fixed_bool4_t  __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 4)));
typedef vbool64_t fixed_bool64_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 64)));

//===----------------------------------------------------------------------===//
// Eligible: single field → coerced to scalable vector
//===----------------------------------------------------------------------===//

struct st_1field { fixed_int32m1_t x; };
// CHECK-LABEL: @test_1field(
// CHECK-SAME: <vscale x 2 x i32>
void test_1field(struct st_1field s) {}

// Return type also coerced
// CHECK: define{{.*}} <vscale x 2 x i32> @test_return_1field(
struct st_1field test_return_1field(struct st_1field s) { return s; }

// double element type
struct st_f64_1field { fixed_float64m1_t x; };
// CHECK-LABEL: @test_f64_1field(
// CHECK-SAME: <vscale x 1 x double>
void test_f64_1field(struct st_f64_1field s) {}

//===----------------------------------------------------------------------===//
// Eligible: single array[1] field → coerced to scalable vector
//===----------------------------------------------------------------------===//

struct st_arr1 { fixed_int32m1_t x[1]; };
// CHECK-LABEL: @test_arr1(
// CHECK-SAME: <vscale x 2 x i32>
void test_arr1(struct st_arr1 s) {}

//===----------------------------------------------------------------------===//
// Eligible: multiple same-type fields → coerced to vector tuple
//===----------------------------------------------------------------------===//

// 2 fields: 2 * LMUL1 = 2 registers
struct st_2field { fixed_int32m1_t x; fixed_int32m1_t y; };
// CHECK-LABEL: @test_2field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 2)
void test_2field(struct st_2field s) {}

// Return type also coerced to tuple
// CHECK: define{{.*}} target("riscv.vector.tuple", <vscale x 8 x i8>, 2) @test_return_2field(
struct st_2field test_return_2field(struct st_2field s) { return s; }

// 3 fields: 3 * LMUL1 = 3 registers
struct st_3field { fixed_int32m1_t a; fixed_int32m1_t b; fixed_int32m1_t c; };
// CHECK-LABEL: @test_3field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 3)
void test_3field(struct st_3field s) {}

// 8 fields: 8 * LMUL1 = 8 registers (at the limit)
struct st_8field {
  fixed_int32m1_t a, b, c, d, e, f, g, h;
};
// CHECK-LABEL: @test_8field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 8)
void test_8field(struct st_8field s) {}

// double element type, 2 fields
struct st_f64_2field { fixed_float64m1_t x; fixed_float64m1_t y; };
// CHECK-LABEL: @test_f64_2field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 2)
void test_f64_2field(struct st_f64_2field s) {}

//===----------------------------------------------------------------------===//
// Eligible: single array[N] field → coerced to vector tuple
//===----------------------------------------------------------------------===//

// array[4]: 4 * LMUL1 = 4 registers
struct st_arr4 { fixed_int32m1_t x[4]; };
// CHECK-LABEL: @test_arr4(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 4)
void test_arr4(struct st_arr4 s) {}

// array[8]: 8 * LMUL1 = 8 registers (at the limit)
struct st_arr8 { fixed_int32m1_t x[8]; };
// CHECK-LABEL: @test_arr8(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 8)
void test_arr8(struct st_arr8 s) {}

//===----------------------------------------------------------------------===//
// Eligible: high-LMUL fields
//===----------------------------------------------------------------------===//

// 2 * LMUL4 = 8 registers (at the limit)
struct st_m4_2field { fixed_int32m4_t x; fixed_int32m4_t y; };
// CHECK-LABEL: @test_m4_2field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 32 x i8>, 2)
void test_m4_2field(struct st_m4_2field s) {}

//===----------------------------------------------------------------------===//
// Fractional LMUL
//===----------------------------------------------------------------------===//

struct st_mf2_1field { fixed_int32mf2_t x; };
// CHECK-LABEL: @test_mf2_1field(
// CHECK-SAME: <vscale x 1 x i32>
void test_mf2_1field(struct st_mf2_1field s) {}

struct st_mf2_2field { fixed_int32mf2_t x; fixed_int32mf2_t y; };
// CHECK-LABEL: @test_mf2_2field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 4 x i8>, 2)
void test_mf2_2field(struct st_mf2_2field s) {}

struct st_mf2_8field { fixed_int32mf2_t a, b, c, d, e, f, g, h; };
// CHECK-LABEL: @test_mf2_8field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 4 x i8>, 8)
void test_mf2_8field(struct st_mf2_8field s) {}

//===----------------------------------------------------------------------===//
// Ineligible: too many fields or array elements → passed indirectly
//===----------------------------------------------------------------------===//

// 9 fields: exceeds max count of 8
struct st_9field {
  fixed_int32m1_t a, b, c, d, e, f, g, h, i;
};
// CHECK-LABEL: @test_9field(
// CHECK-SAME: ptr
void test_9field(struct st_9field s) {}

// array[9]: exceeds max count of 8
struct st_arr9 { fixed_int32m1_t x[9]; };
// CHECK-LABEL: @test_arr9(
// CHECK-SAME: ptr
void test_arr9(struct st_arr9 s) {}

//===----------------------------------------------------------------------===//
// Ineligible: register limit exceeded → passed indirectly
//===----------------------------------------------------------------------===//

// 3 * LMUL4 = 12 registers > 8
struct st_m4_3field { fixed_int32m4_t x; fixed_int32m4_t y; fixed_int32m4_t z; };
// CHECK-LABEL: @test_m4_3field(
// CHECK-SAME: ptr
void test_m4_3field(struct st_m4_3field s) {}

//===----------------------------------------------------------------------===//
// Ineligible: heterogeneous or non-vector fields → passed indirectly
//===----------------------------------------------------------------------===//

// Mixed vector element types
struct st_hetero { fixed_int32m1_t x; fixed_float64m1_t y; };
// CHECK-LABEL: @test_hetero(
// CHECK-SAME: ptr
void test_hetero(struct st_hetero s) {}

// Non-vector field
struct st_mixed { fixed_int32m1_t x; int y; };
// CHECK-LABEL: @test_mixed(
// CHECK-SAME: ptr
void test_mixed(struct st_mixed s) {}

//===----------------------------------------------------------------------===//
// Ineligible: union → passed indirectly
//===----------------------------------------------------------------------===//

union u_vecs { fixed_int32m1_t x; fixed_int32m1_t y; };
// CHECK-LABEL: @test_union(
// CHECK-SAME: ptr
void test_union(union u_vecs u) {}

//===----------------------------------------------------------------------===//
// Ineligible: multiple array fields → passed indirectly
//===----------------------------------------------------------------------===//

struct st_two_arrays { fixed_int32m1_t x[2]; fixed_int32m1_t y[2]; };
// CHECK-LABEL: @test_two_arrays(
// CHECK-SAME: ptr
void test_two_arrays(struct st_two_arrays s) {}

//===----------------------------------------------------------------------===//
// Mask (bool) types: single field → scalable vector
// MinElts = divideCeil(NumElts, VScale) with i8 element type.
//===----------------------------------------------------------------------===//

// fixed_bool1_t: <32 x i8>, MinElts = divideCeil(32,4) = 8 -> <vscale x 8 x i8>
struct st_bool1_1field { fixed_bool1_t x; };
// CHECK-LABEL: @test_bool1_1field(
// CHECK-SAME: <vscale x 8 x i8>
void test_bool1_1field(struct st_bool1_1field s) {}

// fixed_bool4_t: <8 x i8>, MinElts = divideCeil(8,4) = 2 -> <vscale x 2 x i8>
struct st_bool4_1field { fixed_bool4_t x; };
// CHECK-LABEL: @test_bool4_1field(
// CHECK-SAME: <vscale x 2 x i8>
void test_bool4_1field(struct st_bool4_1field s) {}

// fixed_bool64_t: sub-byte <1 x i8>, MinElts = divideCeil(1,4) = 1 -> <vscale x 1 x i8>
struct st_bool64_1field { fixed_bool64_t x; };
// CHECK-LABEL: @test_bool64_1field(
// CHECK-SAME: <vscale x 1 x i8>
void test_bool64_1field(struct st_bool64_1field s) {}

//===----------------------------------------------------------------------===//
// Mask types: multiple fields → vector tuple
// I8EltCount = divideCeil(NumElts * 8, VScale * 8)
//===----------------------------------------------------------------------===//

// 2 * fixed_bool1_t: I8EltCount = divideCeil(32*8, 4*8) = 8
struct st_bool1_2field { fixed_bool1_t x; fixed_bool1_t y; };
// CHECK-LABEL: @test_bool1_2field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 2)
void test_bool1_2field(struct st_bool1_2field s) {}

// 2 * fixed_bool4_t: I8EltCount = divideCeil(8*8, 4*8) = 2
struct st_bool4_2field { fixed_bool4_t x; fixed_bool4_t y; };
// CHECK-LABEL: @test_bool4_2field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 2 x i8>, 2)
void test_bool4_2field(struct st_bool4_2field s) {}

// 2 * fixed_bool64_t (sub-byte): I8EltCount = divideCeil(1*8, 4*8) = 1
struct st_bool64_2field { fixed_bool64_t x; fixed_bool64_t y; };
// CHECK-LABEL: @test_bool64_2field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 1 x i8>, 2)
void test_bool64_2field(struct st_bool64_2field s) {}

// 8 * fixed_bool1_t: 8 * divideCeil(32*8, 4*64) = 8*1 = 8 registers (at limit)
struct st_bool1_8field { fixed_bool1_t a, b, c, d, e, f, g, h; };
// CHECK-LABEL: @test_bool1_8field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 8)
void test_bool1_8field(struct st_bool1_8field s) {}

//===----------------------------------------------------------------------===//
// Mask types: ineligible cases
//===----------------------------------------------------------------------===//

// 9 fields: exceeds count limit of 8
struct st_bool1_9field { fixed_bool1_t a, b, c, d, e, f, g, h, i; };
// CHECK-LABEL: @test_bool1_9field(
// CHECK-SAME: ptr
void test_bool1_9field(struct st_bool1_9field s) {}

// Mixed mask kinds: different canonical types → ineligible
struct st_bool_hetero { fixed_bool1_t x; fixed_bool4_t y; };
// CHECK-LABEL: @test_bool_hetero(
// CHECK-SAME: ptr
void test_bool_hetero(struct st_bool_hetero s) {}

// Mixed mask and data: different kinds → ineligible
struct st_bool_data_mix { fixed_bool1_t x; fixed_int32m1_t y; };
// CHECK-LABEL: @test_bool_data_mix(
// CHECK-SAME: ptr
void test_bool_data_mix(struct st_bool_data_mix s) {}
