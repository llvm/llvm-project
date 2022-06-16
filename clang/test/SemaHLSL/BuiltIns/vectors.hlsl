// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 

#include "hlsl.h"

[numthreads(1,1,1)]
int entry() {
  // verify that the alias is generated inside the hlsl namespace
  hlsl::vector<float, 2> Vec2 = {1.0, 2.0};

  // verify that you don't need to specify the namespace
  vector<float, 2> Vec2a = {1.0, 2.0};

  // verify the typedef works
  uint3 UVec = {1, 2, 3};

  // build a big vector
  vector<float, 4> Vec4 = {1.0, 2.0, 3.0, 4.0};
  // verify swizzles work
  vector<float, 3> Vec3 = Vec4.xyz;
  return 1;
}
