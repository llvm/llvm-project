// RUN: %clang_cc1 -S -emit-llvm -Wno-unknown-pragmas %s -o - | FileCheck %s

// nextUp(1.F) == 0x1.000002p0F

const double _Complex C0 = 0x1.000001p0 + 0x1.000001p0I;

#pragma STDC FENV_ROUND FE_UPWARD

float F1u = 1.0F + 0x0.000002p0F;
float F2u = 1.0F + 0x0.000001p0F;
float F3u = 0x1.000001p0;
// CHECK: @F1u = {{.*}} float 0x3FF0000020000000
// CHECK: @F2u = {{.*}} float 0x3FF0000020000000
// CHECK: @F3u = {{.*}} float 0x3FF0000020000000

float _Complex C1u = C0;
// CHECK: @C1u = {{.*}} { float, float } { float 0x3FF0000020000000, float 0x3FF0000020000000 }


#pragma STDC FENV_ROUND FE_DOWNWARD

float F1d = 1.0F + 0x0.000002p0F;
float F2d = 1.0F + 0x0.000001p0F;
float F3d = 0x1.000001p0;

// CHECK: @F1d = {{.*}} float 0x3FF0000020000000
// CHECK: @F2d = {{.*}} float 1.000000e+00
// CHECK: @F3d = {{.*}} float 1.000000e+00

float _Complex C1d = C0;
// CHECK: @C1d = {{.*}} { float, float } { float 1.000000e+00, float 1.000000e+00 }
