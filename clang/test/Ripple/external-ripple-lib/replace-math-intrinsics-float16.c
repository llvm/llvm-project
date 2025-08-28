// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -g -c -O2 -emit-llvm %S/external_library.c -o %t
// RUN: %clang -g -O2 -fenable-ripple -emit-llvm -S -o - -ffast-math -fripple-lib %t %s | FileCheck %s --implicit-check-not="warning:"

#include <ripple.h>
#include <ripple_math.h>

#define gen_check_unary_mathfn(N, OP, LONGTYPE)                                \
  void check_##OP(const LONGTYPE x[N], LONGTYPE y[N]) {                        \
    ripple_block_t BS = ripple_set_block_shape(0, N);                                               \
    int v0 = ripple_id(BS, 0);                                                  \
    y[v0] = OP(x[v0]);                                                         \
  }

#define gen_check_binary_mathfn(N, OP, LONGTYPE)                               \
  void check_##OP(const LONGTYPE x[N], LONGTYPE y[N], LONGTYPE z[N]) {         \
    ripple_block_t BS = ripple_set_block_shape(0, N);                                               \
    int v0 = ripple_id(BS, 0);                                                  \
    z[v0] = OP(x[v0], y[v0]);                                                  \
  }


// CHECK: @check_sqrtf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_sqrtf16
gen_check_unary_mathfn(128, sqrtf16, _Float16);

// CHECK: @check_asinf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_asinf16
gen_check_unary_mathfn(128, asinf16, _Float16);

// CHECK: @check_acosf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_acosf16
gen_check_unary_mathfn(128, acosf16, _Float16);

// CHECK: @check_atanf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_atanf16
gen_check_unary_mathfn(128, atanf16, _Float16);

// CHECK: @check_atan2f16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_atan2f16
gen_check_binary_mathfn(128, atan2f16, _Float16);

// CHECK: @check_sinf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_sinf16
gen_check_unary_mathfn(128, sinf16, _Float16);

// CHECK: @check_cosf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_cosf16
gen_check_unary_mathfn(128, cosf16, _Float16);

// CHECK: @check_tanf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_tanf16
gen_check_unary_mathfn(128, tanf16, _Float16);

// CHECK: @check_sinhf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_sinhf16
gen_check_unary_mathfn(128, sinhf16, _Float16);

// CHECK: @check_coshf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_coshf16
gen_check_unary_mathfn(128, coshf16, _Float16);

// CHECK: @check_tanhf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_tanhf16
gen_check_unary_mathfn(128, tanhf16, _Float16);

// CHECK: @check_powf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_powf16
gen_check_binary_mathfn(128, powf16, _Float16);

// CHECK: @check_logf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_logf16
gen_check_unary_mathfn(128, logf16, _Float16);

// CHECK: @check_log10f16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_log10f16
gen_check_unary_mathfn(128, log10f16, _Float16);

// CHECK: @check_log2f16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_log2f16
gen_check_unary_mathfn(128, log2f16, _Float16);

// CHECK: @check_expf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_expf16
gen_check_unary_mathfn(128, expf16, _Float16);

// CHECK: @check_exp2f16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_exp2f16
gen_check_unary_mathfn(128, exp2f16, _Float16);

// CHECK: @check_exp10f16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_exp10f16
gen_check_unary_mathfn(128, exp10f16, _Float16);

// CHECK: @check_fabsf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_fabsf16
gen_check_unary_mathfn(128, fabsf16, _Float16);

// CHECK: @check_copysignf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_copysignf16
gen_check_binary_mathfn(128, copysignf16, _Float16);

// CHECK: @check_floorf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_floorf16
gen_check_unary_mathfn(128, floorf16, _Float16);

// CHECK: @check_ceilf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_ceilf16
gen_check_unary_mathfn(128, ceilf16, _Float16);

// CHECK: @check_truncf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_truncf16
gen_check_unary_mathfn(128, truncf16, _Float16);

// CHECK: @check_rintf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_rintf16
gen_check_unary_mathfn(128, rintf16, _Float16);

// CHECK: @check_nearbyintf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_nearbyintf16
gen_check_unary_mathfn(128, nearbyintf16, _Float16);

// CHECK: @check_roundf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_roundf16
gen_check_unary_mathfn(128, roundf16, _Float16);

// CHECK: @check_roundevenf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_roundevenf16
gen_check_unary_mathfn(128, roundevenf16, _Float16);

#define N 128

// CHECK: @check_ldexpf16
// CHECK-COUNT-2: call{{.*}}@ripple_ew_pure_ldexpf16
void check_ldexpf16(const _Float16 x[N], int y[N], _Float16 z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = ldexpf16(x[v0], y[v0]);
}
