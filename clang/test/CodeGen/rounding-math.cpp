// RUN: %clang_cc1 -S -emit-llvm -triple i386-linux -Wno-unknown-pragmas -frounding-math %s -o - | FileCheck %s

constexpr float func_01(float x, float y) {
  return x + y;
}

float V1 = func_01(1.0F, 0x0.000001p0F);
float V2 = 1.0F + 0x0.000001p0F;
float V3 = func_01(1.0F, 2.0F);

// CHECK: @V1 = {{.*}}global float 1.000000e+00, align 4
// CHECK: @V2 = {{.*}}global float 1.000000e+00, align 4
// CHECK: @V3 = {{.*}}global float 3.000000e+00, align 4
