// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -O3 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-vulkan-compute -x hlsl -emit-llvm -O3 -o - %s | FileCheck %s

// All references to unused resources should be removed by optimizations.
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
