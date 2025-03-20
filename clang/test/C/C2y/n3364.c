// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -emit-llvm -o - %s
// RUN: %clang_cc1 -verify -Wall -pedantic -emit-llvm -o - %s
// expected-no-diagnostics

/* WG14 N3364: Yes
 * Give consistent wording for SNAN initialization v3
 *
 * Ensure that initializing from a signaling NAN (optionally with a unary + or
 * -) at translation time behaves correctly at runtime.
 */

#define FLT_SNAN __builtin_nansf("1")
#define DBL_SNAN __builtin_nans("1")
#define LD_SNAN __builtin_nansl("1")

float f1 = FLT_SNAN;
float f2 = +FLT_SNAN;
float f3 = -FLT_SNAN;
// CHECK: @f1 = {{.*}}global float 0x7FF0000020000000
// CHECK: @f2 = {{.*}}global float 0x7FF0000020000000
// CHECK: @f3 = {{.*}}global float 0xFFF0000020000000

double d1 = DBL_SNAN;
double d2 = +DBL_SNAN;
double d3 = -DBL_SNAN;
// CHECK: @d1 = {{.*}}global double 0x7FF0000000000001
// CHECK: @d2 = {{.*}}global double 0x7FF0000000000001
// CHECK: @d3 = {{.*}}global double 0xFFF0000000000001

long double ld1 = LD_SNAN;
long double ld2 = +LD_SNAN;
long double ld3 = -LD_SNAN;
// CHECK: @ld1 = {{.*}}global {{double 0x7FF0000000000001|x86_fp80 0xK7FFF8000000000000001|fp128 0xL00000000000000017FFF000000000000}}
// CHECK: @ld2 = {{.*}}global {{double 0x7FF0000000000001|x86_fp80 0xK7FFF8000000000000001|fp128 0xL00000000000000017FFF000000000000}}
// CHECK: @ld3 = {{.*}}global {{double 0xFFF0000000000001|x86_fp80 0xKFFFF8000000000000001|fp128 0xL0000000000000001FFFF000000000000}}
