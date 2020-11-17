// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=512 -flax-vector-conversions=none -fallow-half-arguments-and-returns -ffreestanding -fsyntax-only -verify=lax-vector-none %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=512 -flax-vector-conversions=integer -fallow-half-arguments-and-returns -ffreestanding -fsyntax-only -verify=lax-vector-integer %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=512 -flax-vector-conversions=all -fallow-half-arguments-and-returns -ffreestanding -fsyntax-only -verify=lax-vector-all %s

// lax-vector-all-no-diagnostics

#include <arm_sve.h>

#define N __ARM_FEATURE_SVE_BITS
#define FIXED_ATTR __attribute__((arm_sve_vector_bits(N)))

typedef svfloat32_t fixed_float32_t FIXED_ATTR;
typedef svint32_t fixed_int32_t FIXED_ATTR;

void allowed_with_integer_lax_conversions() {
  fixed_int32_t fi32;
  svint64_t si64;

  // The implicit cast here should fail if -flax-vector-conversions=none, but pass if
  // -flax-vector-conversions={integer,all}.
  fi32 = si64;
  // lax-vector-none-error@-1 {{assigning to 'fixed_int32_t' (vector of 16 'int' values) from incompatible type}}
  si64 = fi32;
  // lax-vector-none-error@-1 {{assigning to 'svint64_t' (aka '__SVInt64_t') from incompatible type}}
}

void allowed_with_all_lax_conversions() {
  fixed_float32_t ff32;
  svfloat64_t sf64;

  // The implicit cast here should fail if -flax-vector-conversions={none,integer}, but pass if
  // -flax-vector-conversions=all.
  ff32 = sf64;
  // lax-vector-none-error@-1 {{assigning to 'fixed_float32_t' (vector of 16 'float' values) from incompatible type}}
  // lax-vector-integer-error@-2 {{assigning to 'fixed_float32_t' (vector of 16 'float' values) from incompatible type}}
  sf64 = ff32;
  // lax-vector-none-error@-1 {{assigning to 'svfloat64_t' (aka '__SVFloat64_t') from incompatible type}}
  // lax-vector-integer-error@-2 {{assigning to 'svfloat64_t' (aka '__SVFloat64_t') from incompatible type}}
}
