// RUN: %clang_cc1 -S -emit-llvm -triple i386-linux -std=c++2a -Wno-unknown-pragmas %s -o - | FileCheck %s
// RUN: %clang_cc1 -S -emit-llvm -triple i386-linux -fexperimental-new-constant-interpreter -std=c++2a -Wno-unknown-pragmas %s -o - | FileCheck %s


#pragma STDC FENV_ROUND FE_UPWARD

float F1u = 1.0F + 0x0.000002p0F;
float F2u = 1.0F + 0x0.000001p0F;
float F3u = 0x1.000001p0;
// CHECK: @F1u = {{.*}} float 0x3FF0000020000000
// CHECK: @F2u = {{.*}} float 0x3FF0000020000000
// CHECK: @F3u = {{.*}} float 0x3FF0000020000000

float FI1u = 0xFFFFFFFFU;
// CHECK: @FI1u = {{.*}} float 0x41F0000000000000

#pragma STDC FENV_ROUND FE_DOWNWARD

float F1d = 1.0F + 0x0.000002p0F;
float F2d = 1.0F + 0x0.000001p0F;
float F3d = 0x1.000001p0;

// CHECK: @F1d = {{.*}} float 0x3FF0000020000000
// CHECK: @F2d = {{.*}} float 1.000000e+00
// CHECK: @F3d = {{.*}} float 1.000000e+00


float FI1d = 0xFFFFFFFFU;
// CHECK: @FI1d = {{.*}} float 0x41EFFFFFE0000000

// nextUp(1.F) == 0x1.000002p0F

constexpr float add_round_down(float x, float y) {
  #pragma STDC FENV_ROUND FE_DOWNWARD
  float res = x;
  res = res + y;
  return res;
}

constexpr float add_round_up(float x, float y) {
  #pragma STDC FENV_ROUND FE_UPWARD
  float res = x;
  res = res + y;
  return res;
}

float V1 = add_round_down(1.0F, 0x0.000001p0F);
float V2 = add_round_up(1.0F, 0x0.000001p0F);
// CHECK: @V1 = {{.*}} float 1.000000e+00
// CHECK: @V2 = {{.*}} float 0x3FF0000020000000


constexpr float add_cast_round_down(float x, double y) {
  #pragma STDC FENV_ROUND FE_DOWNWARD
  float res = x;
  res += y;
  return res;
}

constexpr float add_cast_round_up(float x, double y) {
  #pragma STDC FENV_ROUND FE_UPWARD
  float res = x;
  res += y;
  return res;
}

float V3 = add_cast_round_down(1.0F, 0x0.000001p0F);
float V4 = add_cast_round_up(1.0F, 0x0.000001p0F);

// CHECK: @V3 = {{.*}} float 1.000000e+00
// CHECK: @V4 = {{.*}} float 0x3FF0000020000000
