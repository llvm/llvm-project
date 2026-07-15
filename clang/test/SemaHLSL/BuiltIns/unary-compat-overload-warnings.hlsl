// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=acos %s 2>&1 | FileCheck %s -DFUNC=acos
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=asin %s 2>&1 | FileCheck %s -DFUNC=asin
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=atan %s 2>&1 | FileCheck %s -DFUNC=atan
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=ceil %s 2>&1 | FileCheck %s -DFUNC=ceil
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=cos %s 2>&1 | FileCheck %s -DFUNC=cos
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=cosh %s 2>&1 | FileCheck %s -DFUNC=cosh
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=degrees %s 2>&1 | FileCheck %s -DFUNC=degrees
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=exp %s 2>&1 | FileCheck %s -DFUNC=exp
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=exp2 %s 2>&1 | FileCheck %s -DFUNC=exp2
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=floor %s 2>&1 | FileCheck %s -DFUNC=floor
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=frac %s 2>&1 | FileCheck %s -DFUNC=frac
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=log %s 2>&1 | FileCheck %s -DFUNC=log
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=log10 %s 2>&1 | FileCheck %s -DFUNC=log10
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=log2 %s 2>&1 | FileCheck %s -DFUNC=log2
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=normalize %s 2>&1 | FileCheck %s -DFUNC=normalize
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=rsqrt %s 2>&1 | FileCheck %s -DFUNC=rsqrt
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=round %s 2>&1 | FileCheck %s -DFUNC=round
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=sin %s 2>&1 | FileCheck %s -DFUNC=sin
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=sinh %s 2>&1 | FileCheck %s -DFUNC=sinh
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=sqrt %s 2>&1 | FileCheck %s -DFUNC=sqrt
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=tan %s 2>&1 | FileCheck %s -DFUNC=tan
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=tanh %s 2>&1 | FileCheck %s -DFUNC=tanh
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=trunc %s 2>&1 | FileCheck %s -DFUNC=trunc
// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes -DFUNC=radians %s 2>&1 | FileCheck %s -DFUNC=radians

// unary double overloads
float test_unary_double(double p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0);
}

float2 test_unary_double2(double2 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0);
}

float3 test_unary_double3(double3 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0);
}

float4 test_unary_double4(double4 p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x 64 bit API lowering for [[FUNC]] is deprecated. Explicitly cast parameters to 32 or 16 bit types.
  return FUNC(p0);
}

// unary integer overloads
// only test scalar ones for brevity
float test_unary_int(int p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0);
}

float test_unary_int(uint p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0);
}

float test_unary_int(int64_t p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0);
}

float test_unary_int(uint64_t p0) {
  // CHECK: warning: '[[FUNC]]' is deprecated: In 202x int lowering for [[FUNC]] is deprecated. Explicitly cast parameters to float types.
  return FUNC(p0);
}
