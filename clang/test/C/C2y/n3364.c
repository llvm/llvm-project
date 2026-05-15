// RUN: %clang_cc1 -verify -std=c2y -ffreestanding -Wall -pedantic -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -verify -ffreestanding -Wall -pedantic -emit-llvm -o - %s | FileCheck %s
// expected-no-diagnostics

/* WG14 N3364: Yes
 * Give consistent wording for SNAN initialization v3
 *
 * Ensure that initializing from a signaling NAN (optionally with a unary + or
 * -) at translation time behaves correctly at runtime.
 *
 * This also serves as a test for C23's WG14 N2710 which introduces these
 * macros into float.h in Clang 22.
 */

#if __STDC_VERSION__ >= 202311L
#include <float.h>
#else
#define FLT_SNAN __builtin_nansf("")
#define DBL_SNAN __builtin_nans("")
#define LDBL_SNAN __builtin_nansl("")
#endif

float f1 = FLT_SNAN;
float f2 = +FLT_SNAN;
float f3 = -FLT_SNAN;
// CHECK: @f1 = {{.*}}global float +snan(0x200000)
// CHECK: @f2 = {{.*}}global float +snan(0x200000)
// CHECK: @f3 = {{.*}}global float -snan(0x200000)

double d1 = DBL_SNAN;
double d2 = +DBL_SNAN;
double d3 = -DBL_SNAN;
// CHECK: @d1 = {{.*}}global double +snan(0x4000000000000)
// CHECK: @d2 = {{.*}}global double +snan(0x4000000000000)
// CHECK: @d3 = {{.*}}global double -snan(0x4000000000000)

long double ld1 = LDBL_SNAN;
long double ld2 = +LDBL_SNAN;
long double ld3 = -LDBL_SNAN;
// CHECK: @ld1 = {{.*}}global {{double|x86_fp80|fp128|ppc_fp128}} +snan(
// CHECK: @ld2 = {{.*}}global {{double|x86_fp80|fp128|ppc_fp128}} +snan(
// CHECK: @ld3 = {{.*}}global {{double|x86_fp80|fp128|ppc_fp128}} -snan(
