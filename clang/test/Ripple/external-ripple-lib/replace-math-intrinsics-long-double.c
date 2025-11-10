// REQUIRES: target-x86_64
// RUN: %clang -c -O2 -emit-llvm %S/external_library.c -o %t
// RUN: %clang -O2 -fenable-ripple -emit-llvm -S -o - -ffast-math -fripple-lib %t -mllvm -ripple-disable-link %s | FileCheck %s --implicit-check-not="warning:"

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


// CHECK: @check_sqrtl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_sqrtl
gen_check_unary_mathfn(128, sqrtl, long double);

// CHECK: @check_asinl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_asinl
gen_check_unary_mathfn(128, asinl, long double);

// CHECK: @check_acosl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_acosl
gen_check_unary_mathfn(128, acosl, long double);

// CHECK: @check_atanl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_atanl
gen_check_unary_mathfn(128, atanl, long double);

// CHECK: @check_atan2l
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_atan2l
gen_check_binary_mathfn(128, atan2l, long double);

// CHECK: @check_sinl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_sinl
gen_check_unary_mathfn(128, sinl, long double);

// CHECK: @check_cosl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_cosl
gen_check_unary_mathfn(128, cosl, long double);

// CHECK: @check_tanl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_tanl
gen_check_unary_mathfn(128, tanl, long double);

// CHECK: @check_sinhl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_sinhl
gen_check_unary_mathfn(128, sinhl, long double);

// CHECK: @check_coshl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_coshl
gen_check_unary_mathfn(128, coshl, long double);

// CHECK: @check_tanhl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_tanhl
gen_check_unary_mathfn(128, tanhl, long double);

// CHECK: @check_powl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_powl
gen_check_binary_mathfn(128, powl, long double);

// CHECK: @check_logl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_logl
gen_check_unary_mathfn(128, logl, long double);

// CHECK: @check_log10l
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_log10l
gen_check_unary_mathfn(128, log10l, long double);

// CHECK: @check_log2l
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_log2l
gen_check_unary_mathfn(128, log2l, long double);

// CHECK: @check_expl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_expl
gen_check_unary_mathfn(128, expl, long double);

// CHECK: @check_exp2l
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_exp2l
gen_check_unary_mathfn(128, exp2l, long double);

// CHECK: @check_exp10l
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_exp10l
gen_check_unary_mathfn(128, exp10l, long double);

// CHECK: @check_fabsl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_fabsl
gen_check_unary_mathfn(128, fabsl, long double);

// CHECK: @check_copysignl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_copysignl
gen_check_binary_mathfn(128, copysignl, long double);

// CHECK: @check_floorl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_floorl
gen_check_unary_mathfn(128, floorl, long double);

// CHECK: @check_ceill
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_ceill
gen_check_unary_mathfn(128, ceill, long double);

// CHECK: @check_truncl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_truncl
gen_check_unary_mathfn(128, truncl, long double);

// CHECK: @check_rintl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_rintl
gen_check_unary_mathfn(128, rintl, long double);

// CHECK: @check_nearbyintl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_nearbyintl
gen_check_unary_mathfn(128, nearbyintl, long double);

// CHECK: @check_roundl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_roundl
gen_check_unary_mathfn(128, roundl, long double);

// CHECK: @check_roundevenl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_roundevenl
gen_check_unary_mathfn(128, roundevenl, long double);

#define N 128

// CHECK: @check_ldexpl
// CHECK-COUNT-16: call{{.*}}@ripple_ew_pure_ldexpl
void check_ldexpl(const long double x[N], int y[N], long double z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = ldexpl(x[v0], y[v0]);
}
