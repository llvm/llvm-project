// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -fopenmp      -x c -emit-llvm %s -o - -femit-all-decls | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -fopenmp-simd -x c -emit-llvm %s -o - -femit-all-decls | FileCheck %s

// REQUIRES: aarch64-registered-target
// Note: -fopemp and -fopenmp-simd behavior are expected to be the same.

// This test checks the values of Narrowest Data Size (NDS), as defined in
// https://github.com/ARM-software/abi-aa/tree/main/vfabia64
//
// NDS is used to compute the <vlen> token in the name of AdvSIMD
// vector functions when no `simdlen` is specified, with the rule:
//
// if NDS(f) = 1, then VLEN = 16, 8;
// if NDS(f) = 2, then VLEN = 8, 4;
// if NDS(f) = 4, then VLEN = 4, 2;
// if NDS(f) = 8 or NDS(f) = 16, then VLEN = 2.

// NDS(NDS_is_sizeof_char) = 1
#pragma omp declare simd notinbranch
char NDS_is_sizeof_char(short in);
// CHECK-DAG: _ZGVnN16v_NDS_is_sizeof_char
// CHECK-DAG: _ZGVnN8v_NDS_is_sizeof_char
// CHECK-NOT: _ZGV{{.*}}_NDS_is_sizeof_char

// NDS(NDS_is_sizeof_short) = 2
#pragma omp declare simd notinbranch
int NDS_is_sizeof_short(short in);
// CHECK-DAG: _ZGVnN8v_NDS_is_sizeof_short
// CHECK-DAG: _ZGVnN4v_NDS_is_sizeof_short
// CHECK-NOT: _ZGV{{.*}}_NDS_is_sizeof_short

// NDS(NDS_is_sizeof_float_with_linear) = 4, and not 2, because the pointers are
// marked as `linear` and therefore the size of the pointee realizes
// the NDS.
#pragma omp declare simd linear(sin) notinbranch
void NDS_is_sizeof_float_with_linear(double in, float *sin);
// Neon accepts only power of 2 values as <vlen>.
// CHECK-DAG: _ZGVnN4vl4_NDS_is_sizeof_float_with_linear
// CHECK-DAG: _ZGVnN2vl4_NDS_is_sizeof_float_with_linear
// CHECK-NOT: _ZGV{{.*}}_NDS_is_sizeof_float_with_linear

// NDS(NDS_is_size_of_float) = 4
#pragma omp declare simd notinbranch
double NDS_is_size_of_float(float in);
// CHECK-DAG: _ZGVnN4v_NDS_is_size_of_float
// CHECK-DAG: _ZGVnN2v_NDS_is_size_of_float
// CHECK-NOT: _ZGV{{.*}}_NDS_is_size_of_float

// NDS(NDS_is_sizeof_double) = 8
#pragma omp declare simd linear(sin) notinbranch
void NDS_is_sizeof_double(double in, double *sin);
// CHECK-DAG: _ZGVnN2vl8_NDS_is_sizeof_double
// CHECK-NOT: _ZGV{{.*}}_NDS_is_sizeof_double

// NDS(double_complex) = 16
#pragma omp declare simd notinbranch
double _Complex double_complex(double _Complex);
// CHECK-DAG: _ZGVnN2v_double_complex
// CHECK-NOT: _ZGV{{.*}}_double_complex

// NDS(double_complex_linear_char) = 1, becasue `x` is marked linear.
#pragma omp declare simd linear(x) notinbranch
double _Complex double_complex_linear_char(double _Complex y, char *x);
// CHECK-DAG: _ZGVnN8vl_double_complex_linear_char
// CHECK-DAG: _ZGVnN16vl_double_complex_linear_char
// CHECK-NOT: _ZGV{{.*}}_double_complex_linear_char

static float *F;
static double *D;
static short S;
static int I;
static char C;
static double _Complex DC;
void do_something() {
  C = NDS_is_sizeof_char(S);
  I = NDS_is_sizeof_short(S);
  NDS_is_sizeof_float_with_linear(*D, F);
  *D = NDS_is_size_of_float(*F);
  NDS_is_sizeof_double(*D, D);
  DC = double_complex(DC);
  DC = double_complex_linear_char(DC, &C);
}
