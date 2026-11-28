// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-library -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=lerp %s 2>&1 | FileCheck %s -DFUNC=lerp

using namespace hlsl;

// ternary double overloads
float test_ternary_double(double p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0, p0);
}

float2 test_ternary_double2(double2 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0, p0);
}

float3 test_ternary_double3(double3 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0, p0);
}

float4 test_ternary_double4(double4 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0, p0);
}

float4x4 test_ternary_double4x4(double4x4 p0) {
  // CHECK: warning: '[[FUNC]]<4U, 4U>' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0, p0, p0);
}

// ternary integer overloads
// only test scalar and matrix ones for brevity
float test_ternary_int(int p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0, p0);
}

float test_ternary_uint(uint p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0, p0);
}

float test_ternary_int64_t(int64_t p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0, p0);
}

float test_ternary_uint64_t(uint64_t p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0, p0);
}

float4x4 test_ternary_int4x4(int4x4 p0) {
  // CHECK: warning: '[[FUNC]]<4U, 4U>' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0, p0);
}

float4x4 test_ternary_uint4x4(uint4x4 p0) {
  // CHECK: warning: '[[FUNC]]<4U, 4U>' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0, p0);
}

float4x4 test_ternary_int64_t4x4(int64_t4x4 p0) {
  // CHECK: warning: '[[FUNC]]<4U, 4U>' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0, p0);
}

float4x4 test_ternary_uint64_t4x4(uint64_t4x4 p0) {
  // CHECK: warning: '[[FUNC]]<4U, 4U>' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0, p0, p0);
}
