// RUN: %clang_dxc -Tlib_6_7 -fcgl  -Fo - %s | FileCheck %s --check-prefix=FLOAT
// RUN: %clang_dxc -Tlib_6_7 -enable-16bit-types -fcgl  -Fo - %s | FileCheck %s --check-prefix=HALF

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
