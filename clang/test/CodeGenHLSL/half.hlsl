// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=HALF

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=FLOAT


// Make sure use float when not enable-16bit-types.
// FLOAT:define {{.*}}float @"?foo@@YA$halff@$halff@0@Z"(float{{[^,]+}}, float{{[^,)]+}})
// FLOAT-NOT:half
// FLOAT:ret float %

// Make sure use half when enable-16bit-types.
// HALF:define {{.*}}half @"?foo@@YA$f16@$f16@0@Z"(half{{[^,]+}}, half{{[^,)]+}})
// HALF-NOT:float
// HALF:ret half %
half foo(half a, half b) {
  return a+b;
}
