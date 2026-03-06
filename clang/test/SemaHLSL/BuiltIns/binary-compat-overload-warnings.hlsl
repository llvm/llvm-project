// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=atan2 %s 2>&1 | FileCheck %s -DFUNC=atan2
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=pow %s 2>&1 | FileCheck %s -DFUNC=pow
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=step %s 2>&1 | FileCheck %s -DFUNC=step

// binary double overloads
float test_binary_double(double p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0);
}

float2 test_binary_double2(double2 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0);
}

float3 test_binary_double3(double3 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0);
}

float4 test_binary_double4(double4 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0);
}

// binary integer overloads
// only test scalar ones for brevity
float test_binary_int(int p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0);
}

float test_binary_int(uint p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0);
}

float test_binary_int(int64_t p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0);
}

float test_binary_int(uint64_t p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0);
}
