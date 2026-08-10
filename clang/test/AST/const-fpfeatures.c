// RUN: %clang_cc1 -emit-llvm -triple i386-linux -Wno-unknown-pragmas %s -o - | FileCheck %s

// nextUp(1.F) == 0x1.000002p0F

const double _Complex C0 = 0x1.000001p0 + 0x1.000001p0I;

#pragma STDC FENV_ROUND FE_UPWARD

float F1u = 1.0F + 0x0.000002p0F;
float F2u = 1.0F + 0x0.000001p0F;
float F3u = 0x1.000001p0;
// CHECK: @F1u = {{.*}} float f0x3F800001
// CHECK: @F2u = {{.*}} float f0x3F800001
// CHECK: @F3u = {{.*}} float f0x3F800001

float FI1u = 0xFFFFFFFFU;
// CHECK: @FI1u = {{.*}} float f0x4F800000

float _Complex C1u = C0;
// CHECK: @C1u = {{.*}} { float, float } { float f0x3F800001, float f0x3F800001 }

float FLu = 0.1F;
// CHECK: @FLu = {{.*}} float 1.000000e-01

typedef float  vector2float  __attribute__((__vector_size__(8)));
typedef double vector2double  __attribute__((__vector_size__(16)));
const vector2float V2Fu = {1.0F + 0x0.000001p0F, 1.0F + 0x0.000002p0F};
vector2double V2Du = __builtin_convertvector(V2Fu, vector2double);
// CHECK: @V2Fu = {{.*}} <2 x float> splat (float f0x3F800001)
// CHECK: @V2Du = {{.*}} <2 x double> splat (double f0x3FF0000020000000)

#pragma STDC FENV_ROUND FE_DOWNWARD

float F1d = 1.0F + 0x0.000002p0F;
float F2d = 1.0F + 0x0.000001p0F;
float F3d = 0x1.000001p0;

// CHECK: @F1d = {{.*}} float f0x3F800001
// CHECK: @F2d = {{.*}} float 1.000000e+00
// CHECK: @F3d = {{.*}} float 1.000000e+00

float FI1d = 0xFFFFFFFFU;
// CHECK: @FI1d = {{.*}} float f0x4F7FFFFF

float _Complex C1d = C0;
// CHECK: @C1d = {{.*}} { float, float } { float 1.000000e+00, float 1.000000e+00 }

float FLd = 0.1F;
// CHECK: @FLd = {{.*}} float f0x3DCCCCCC

const vector2float V2Fd = {1.0F + 0x0.000001p0F, 1.0F + 0x0.000002p0F};
vector2double V2Dd = __builtin_convertvector(V2Fd, vector2double);
// CHECK: @V2Fd = {{.*}} <2 x float> <float 1.000000e+00, float f0x3F800001>
// CHECK: @V2Dd = {{.*}} <2 x double> <double 1.000000e+00, double f0x3FF0000020000000>
