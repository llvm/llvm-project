// Verify that a riscv_rvv_vector_bits typedef keeps the LMUL of the sizeless
// RVV type it was written against. Two typedefs can hold the same number of lanes of the same element type while
// coming from different LMULs, and they must stay distinct types.

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -ffreestanding -fsyntax-only -verify=expected -mvscale-min=2 \
// RUN:   -mvscale-max=2 %s

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -ffreestanding -fsyntax-only -verify=expected,nolax -mvscale-min=2 \
// RUN:   -mvscale-max=2 \
// RUN:   -flax-vector-conversions=none %s

// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

// VLEN=128: vint32m1_t holds 4 i32 lanes at full size, vint32mf2_t holds 2,
// vint32m2_t holds 8. All three typedefs below hold 2 i32 lanes.
typedef vint32m1_t  m1_64  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32mf2_t mf2_64 __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m2_t  m2_64  __attribute__((riscv_rvv_vector_bits(64)));

//===----------------------------------------------------------------------===//
// Same element type and lane count, different LMUL => distinct types.
//
// They are still the same size, so a lax vector conversion allows a direct
// assignment between them as a reinterpret, exactly as it does for GNU
// vectors. -flax-vector-conversions defaults to "all", so this is on unless
// asked otherwise; turning it off shows that the types really are distinct.
//===----------------------------------------------------------------------===//

void distinct_types(void) {
  m1_64 a;
  mf2_64 b;
  m2_64 c;

  a = b; // nolax-error {{assigning to 'm1_64' (vector of 2 'int' values) from incompatible type 'mf2_64'}}
  b = a; // nolax-error {{assigning to 'mf2_64' (vector of 2 'int' values) from incompatible type 'm1_64'}}
  a = c; // nolax-error {{assigning to 'm1_64' (vector of 2 'int' values) from incompatible type 'm2_64'}}
}

//===----------------------------------------------------------------------===//
// Conversion to and from the sizeless type is only allowed for the matching
// LMUL. This goes through areCompatibleRVVTypes, so it is rejected regardless
// of the lax vector conversion setting.
//===----------------------------------------------------------------------===//

void to_sizeless(void) {
  m1_64 a;

  vint32m1_t ok = a;
  (void)ok;

  vint32mf2_t narrower = a; // expected-error {{initializing 'vint32mf2_t' (aka '__rvv_int32mf2_t') with an expression of incompatible type 'm1_64'}}
  (void)narrower;

  vint32m2_t wider = a; // expected-error {{initializing 'vint32m2_t' (aka '__rvv_int32m2_t') with an expression of incompatible type 'm1_64'}}
  (void)wider;
}

void from_sizeless(vint32m1_t v1, vint32mf2_t vh, vint32m2_t v2) {
  m1_64 ok = v1;
  (void)ok;

  m1_64 bad_narrower = vh; // expected-error {{with an expression of incompatible type 'vint32mf2_t' (aka '__rvv_int32mf2_t')}}
  (void)bad_narrower;

  m1_64 bad_wider = v2; // expected-error {{with an expression of incompatible type 'vint32m2_t' (aka '__rvv_int32m2_t')}}
  (void)bad_wider;
}

//===----------------------------------------------------------------------===//
// Calling RVV intrinsics: only the intrinsic of the matching LMUL accepts a
// shorter typedef.
//===----------------------------------------------------------------------===//

void intrinsics_m1(m1_64 a, m1_64 b) {
  // Matching LMUL: fine.
  m1_64 sum = __riscv_vadd_vv_i32m1(a, b, 2);
  (void)sum;

  // Fewer lanes than the typedef's LMUL provides.
  (void)__riscv_vadd_vv_i32mf2(a, b, 2); // expected-error {{passing 'm1_64' (vector of 2 'int' values) to parameter of incompatible type '__rvv_int32mf2_t'}} \
                                         // expected-note {{passing argument to parameter here}}

  // More lanes than the typedef's LMUL provides.
  (void)__riscv_vadd_vv_i32m2(a, b, 2); // expected-error {{passing 'm1_64' (vector of 2 'int' values) to parameter of incompatible type '__rvv_int32m2_t'}} \
                                        // expected-note {{passing argument to parameter here}}
}

void intrinsics_mf2(mf2_64 a, mf2_64 b) {
  mf2_64 sum = __riscv_vadd_vv_i32mf2(a, b, 2);
  (void)sum;

  (void)__riscv_vadd_vv_i32m1(a, b, 2); // expected-error {{passing 'mf2_64' (vector of 2 'int' values) to parameter of incompatible type '__rvv_int32m1_t'}} \
                                        // expected-note {{passing argument to parameter here}}
}

// Loads and stores of the matching LMUL are fine.
void intrinsics_mem(const int *p, int *q) {
  m1_64 v = __riscv_vle32_v_i32m1(p, 2);
  __riscv_vse32_v_i32m1(q, v, 2);
}

//===----------------------------------------------------------------------===//
// A full-size typedef behaves the same way: LMUL comes from the typedef, not
// from the lane count.
//===----------------------------------------------------------------------===//

typedef vint32m1_t m1_128 __attribute__((riscv_rvv_vector_bits(128)));

// The mf2 type cannot even hold 128 bits at this VLEN.
typedef vint32mf2_t mf2_128 __attribute__((riscv_rvv_vector_bits(128))); // expected-error {{invalid RVV vector size '128', must not exceed '64' based on LMUL of type and the minimum VLEN}}

void full_size(void) {
  m1_128 a;
  vint32m1_t ok = a;
  (void)ok;

  vint32m2_t bad = a; // expected-error {{initializing 'vint32m2_t' (aka '__rvv_int32m2_t') with an expression of incompatible type 'm1_128'}}
  (void)bad;
}

//===----------------------------------------------------------------------===//
// Mask typedefs keep the ratio of the vbool<R>_t type they were written
// against in the same way. A shorter mask has fewer lanes than VLEN / R, so
// the ratio cannot be recovered from the lane count.
//
// VLEN=128: vbool4_t covers 32 lanes, vbool8_t 16, vbool32_t 4.
//===----------------------------------------------------------------------===//

typedef vbool4_t  b4_32 __attribute__((riscv_rvv_vector_bits(32)));
typedef vbool8_t  b8_16 __attribute__((riscv_rvv_vector_bits(16)));
typedef vbool32_t b32_4 __attribute__((riscv_rvv_vector_bits(4)));

// Shorter masks.
typedef vbool4_t  b4_16 __attribute__((riscv_rvv_vector_bits(16)));
typedef vbool8_t  b8_8  __attribute__((riscv_rvv_vector_bits(8)));
typedef vbool4_t  b4_8  __attribute__((riscv_rvv_vector_bits(8)));
typedef vbool32_t b32_2 __attribute__((riscv_rvv_vector_bits(2)));
typedef vbool32_t b32_1 __attribute__((riscv_rvv_vector_bits(1)));

void mask_conversions(void) {
  b4_32 a;

  vbool4_t ok = a;
  (void)ok;

  // Only the mask type of the matching ratio is compatible.
  vbool8_t bad = a; // expected-error {{initializing 'vbool8_t' (aka '__rvv_bool8_t') with an expression of incompatible type 'b4_32'}}
  (void)bad;
}

// Different ratios have different lane counts, so they are distinct types
// regardless of the lax vector conversion setting.
void mask_distinct(void) {
  b4_32 a;
  b8_16 b;
  a = b; // expected-error {{assigning to 'b4_32' (vector of 4 'unsigned char' values) from incompatible type 'b8_16' (vector of 2 'unsigned char' values)}}
}

void mask_intrinsics(b4_32 a, b4_32 b) {
  b4_32 r = __riscv_vmand_mm_b4(a, b, 4);
  (void)r;

  (void)__riscv_vmand_mm_b8(a, b, 4); // expected-error {{passing 'b4_32' (vector of 4 'unsigned char' values) to parameter of incompatible type '__rvv_bool8_t'}} \
                                      // expected-note {{passing argument to parameter here}}
}

// A shorter mask converts to and from its own vbool<R>_t type only.
void short_mask_conversions(void) {
  b4_16 a;

  vbool4_t ok = a;
  a = ok;

  vbool8_t bad = a; // expected-error {{initializing 'vbool8_t' (aka '__rvv_bool8_t') with an expression of incompatible type 'b4_16'}}
  (void)bad;
}

// Same lane count, different ratio: distinct types. b8_8 and b4_8 both hold 8
// lanes in one byte, so like the data types above they only convert through
// lax vector conversions.
void short_mask_distinct(void) {
  b8_8 a;
  b4_8 b;
  a = b; // nolax-error {{assigning to 'b8_8' (vector of 1 'unsigned char' value) from incompatible type 'b4_8' (vector of 1 'unsigned char' value)}}
}

// Masks that fit in one byte are distinct by their lane count too.
void sub_byte_mask_distinct(void) {
  b32_2 a;
  b32_1 b;
  a = b; // nolax-error {{assigning to 'b32_2' (vector of 1 'unsigned char' value) from incompatible type 'b32_1' (vector of 1 'unsigned char' value)}}
}

// A mask typedef pairs with a data typedef through the ratio matching the data
// type's LMUL and SEW (vint32m1_t -> vbool32_t). A shorter data typedef
// pairs with a mask of the same lane count.
void mask_with_short_data(m1_64 a, m1_64 b) {
  b32_2 m = __riscv_vmseq_vv_i32m1_b32(a, b, 2);
  m1_64 r = __riscv_vadd_vv_i32m1_m(m, a, b, 2);
  (void)r;
}
