// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve  -fopenmp      -x c -emit-llvm %s -o - -femit-all-decls | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve  -fopenmp-simd -x c -emit-llvm %s -o - -femit-all-decls | FileCheck %s

// REQUIRES: aarch64-registered-target
// Note: -fopemp and -fopenmp-simd behavior are expected to be the same.

// This test checks the values of Widest Data Size (WDS), as defined
// in https://github.com/ARM-software/abi-aa/tree/main/vfabia64
//
// WDS is used to check the accepted values <N> of `simdlen(<N>)` when
// targeting fixed-length SVE vector function names. The values of
// `<N>` that are accepted are such that for X = WDS * <N> * 8,
// 128-bit <= X <= 2048-bit and X is a multiple of 128-bit.

#pragma omp declare simd simdlen(8)
#pragma omp declare simd simdlen(16)
#pragma omp declare simd simdlen(256)
#pragma omp declare simd simdlen(272)
char WDS_is_sizeof_char(char in);
// WDS = 1, simdlen(8) and simdlen(272) are not generated.
// CHECK-DAG: _ZGVsM16v_WDS_is_sizeof_char
// CHECK-DAG: _ZGVsM256v_WDS_is_sizeof_char
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_char

#pragma omp declare simd simdlen(4)
#pragma omp declare simd simdlen(8)
#pragma omp declare simd simdlen(128)
#pragma omp declare simd simdlen(136)
char WDS_is_sizeof_short(short in);
// WDS = 2, simdlen(4) and simdlen(136) are not generated.
// CHECK-DAG: _ZGVsM8v_WDS_is_sizeof_short
// CHECK-DAG: _ZGVsM128v_WDS_is_sizeof_short
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_short

#pragma omp declare simd linear(sin) notinbranch simdlen(2)
#pragma omp declare simd linear(sin) notinbranch simdlen(4)
#pragma omp declare simd linear(sin) notinbranch simdlen(64)
#pragma omp declare simd linear(sin) notinbranch simdlen(68)
void WDS_is_sizeof_float_pointee(float in, float *sin);
// WDS = 4, simdlen(2) and simdlen(68) are not generated.
// CHECK-DAG: _ZGVsM4vl4_WDS_is_sizeof_float_pointee
// CHECK-DAG: _ZGVsM64vl4_WDS_is_sizeof_float_pointee
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_float_pointee

#pragma omp declare simd linear(sin) notinbranch simdlen(2)
#pragma omp declare simd linear(sin) notinbranch simdlen(4)
#pragma omp declare simd linear(sin) notinbranch simdlen(32)
#pragma omp declare simd linear(sin) notinbranch simdlen(34)
void WDS_is_sizeof_double_pointee(float in, double *sin);
// WDS = 8 because of the linear clause, simdlen(34) is not generated.
// CHECK-DAG: _ZGVsM2vl8_WDS_is_sizeof_double_pointee
// CHECK-DAG: _ZGVsM4vl8_WDS_is_sizeof_double_pointee
// CHECK-DAG: _ZGVsM32vl8_WDS_is_sizeof_double_pointee
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_double_pointee

#pragma omp declare simd simdlen(2)
#pragma omp declare simd simdlen(4)
#pragma omp declare simd simdlen(32)
#pragma omp declare simd simdlen(34)
double WDS_is_sizeof_double(double in);
// WDS = 8, simdlen(34) is not generated.
// CHECK-DAG: _ZGVsM2v_WDS_is_sizeof_double
// CHECK-DAG: _ZGVsM4v_WDS_is_sizeof_double
// CHECK-DAG: _ZGVsM32v_WDS_is_sizeof_double
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_double

static char C;
static short S;
static float F;
static double D;

void do_something() {
  C = WDS_is_sizeof_char(C);
  C = WDS_is_sizeof_short(S);
  WDS_is_sizeof_float_pointee(F, &F);
  WDS_is_sizeof_double_pointee(F, &D);
  D = WDS_is_sizeof_double(D);
}
