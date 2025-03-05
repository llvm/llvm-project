<<<<<<< HEAD
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -O3 -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-compute -x hlsl -emit-llvm -O3 -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV
// CHECK-SPIRV: %"class.hlsl::RWBuffer" = type { target("spirv.Image", float, 5, 2, 0, 0, 2, 0) }
// CHECK-DXIL:  %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
=======
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -O3 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-vulkan-compute -x hlsl -emit-llvm -O3 -o - %s | FileCheck %s

// All referenced to an unused resource should be removed by optimizations.
>>>>>>> 46236f4c3dbe11e14fe7ac1f4b903637efedfecf
RWBuffer<float> Buf : register(u5, space3);

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
// CHECK-NOT: resource.handlefrombinding
// CHECK: define void @main()
// CHECK-NEXT: entry:
// CHECK-NEXT: ret void
// CHECK-NOT: resource.handlefrombinding
}
