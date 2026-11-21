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
