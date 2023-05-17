// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=8 -mvscale-max=8 -flax-vector-conversions=none -ffreestanding -fsyntax-only -verify=lax-vector-none %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=8 -mvscale-max=8 -flax-vector-conversions=integer -ffreestanding -fsyntax-only -verify=lax-vector-integer %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=8 -mvscale-max=8 -flax-vector-conversions=all -ffreestanding -fsyntax-only -verify=lax-vector-all %s

// lax-vector-all-no-diagnostics

// REQUIRES: riscv-registered-target

#define RVV_FIXED_ATTR __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)))
#define GNU_FIXED_ATTR __attribute__((vector_size(__riscv_v_fixed_vlen / 8)))

typedef __rvv_int8m1_t vint8m1_t;
typedef __rvv_uint8m1_t vuint8m1_t;
typedef __rvv_int16m1_t vint16m1_t;
typedef __rvv_uint16m1_t vuint16m1_t;
typedef __rvv_int32m1_t vint32m1_t;
typedef __rvv_uint32m1_t vuint32m1_t;
typedef __rvv_int64m1_t vint64m1_t;
typedef __rvv_uint64m1_t vuint64m1_t;
typedef __rvv_float32m1_t vfloat32m1_t;
typedef __rvv_float64m1_t vfloat64m1_t;

typedef vfloat32m1_t rvv_fixed_float32m1_t RVV_FIXED_ATTR;
typedef vint32m1_t rvv_fixed_int32m1_t RVV_FIXED_ATTR;
typedef float gnu_fixed_float32m1_t GNU_FIXED_ATTR;
typedef int gnu_fixed_int32m1_t GNU_FIXED_ATTR;

void rvv_allowed_with_integer_lax_conversions() {
  rvv_fixed_int32m1_t fi32;
  vint64m1_t si64;

  // The implicit cast here should fail if -flax-vector-conversions=none, but pass if
  // -flax-vector-conversions={integer,all}.
  fi32 = si64;
  // lax-vector-none-error@-1 {{assigning to 'rvv_fixed_int32m1_t' (vector of 16 'int' values) from incompatible type}}
  si64 = fi32;
  // lax-vector-none-error@-1 {{assigning to 'vint64m1_t' (aka '__rvv_int64m1_t') from incompatible type}}
}

void rvv_allowed_with_all_lax_conversions() {
  rvv_fixed_float32m1_t ff32;
  vfloat64m1_t sf64;

  // The implicit cast here should fail if -flax-vector-conversions={none,integer}, but pass if
  // -flax-vector-conversions=all.
  ff32 = sf64;
  // lax-vector-none-error@-1 {{assigning to 'rvv_fixed_float32m1_t' (vector of 16 'float' values) from incompatible type}}
  // lax-vector-integer-error@-2 {{assigning to 'rvv_fixed_float32m1_t' (vector of 16 'float' values) from incompatible type}}
  sf64 = ff32;
  // lax-vector-none-error@-1 {{assigning to 'vfloat64m1_t' (aka '__rvv_float64m1_t') from incompatible type}}
  // lax-vector-integer-error@-2 {{assigning to 'vfloat64m1_t' (aka '__rvv_float64m1_t') from incompatible type}}
}

void gnu_allowed_with_integer_lax_conversions() {
  gnu_fixed_int32m1_t fi32;
  vint64m1_t si64;

  // The implicit cast here should fail if -flax-vector-conversions=none, but pass if
  // -flax-vector-conversions={integer,all}.
  fi32 = si64;
  // lax-vector-none-error@-1 {{assigning to 'gnu_fixed_int32m1_t' (vector of 16 'int' values) from incompatible type}}
  si64 = fi32;
  // lax-vector-none-error@-1 {{assigning to 'vint64m1_t' (aka '__rvv_int64m1_t') from incompatible type}}
}

void gnu_allowed_with_all_lax_conversions() {
  gnu_fixed_float32m1_t ff32;
  vfloat64m1_t sf64;

  // The implicit cast here should fail if -flax-vector-conversions={none,integer}, but pass if
  // -flax-vector-conversions=all.
  ff32 = sf64;
  // lax-vector-none-error@-1 {{assigning to 'gnu_fixed_float32m1_t' (vector of 16 'float' values) from incompatible type}}
  // lax-vector-integer-error@-2 {{assigning to 'gnu_fixed_float32m1_t' (vector of 16 'float' values) from incompatible type}}
  sf64 = ff32;
  // lax-vector-none-error@-1 {{assigning to 'vfloat64m1_t' (aka '__rvv_float64m1_t') from incompatible type}}
  // lax-vector-integer-error@-2 {{assigning to 'vfloat64m1_t' (aka '__rvv_float64m1_t') from incompatible type}}
}
