// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -O3 -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-compute -x hlsl -emit-llvm -O3 -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// CHECK-SPIRV: %"class.hlsl::RWBuffer" = type { target("spirv.Image", float, 5, 2, 0, 0, 2, 0) }
// CHECK-DXIL:  %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
RWBuffer<float> Buf : register(u5, space3);

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
// CHECK: define void @main()
// CHECK-NEXT: entry:

// CHECK-SPIRV-NEXT: %[[HANDLE:.*]] = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
// CHECK-SPIRV-NEXT: store target("spirv.Image", float, 5, 2, 0, 0, 2, 0) %[[HANDLE:.*]], ptr @Buf, align 8

// CHECK-DXIL-NEXT: %[[HANDLE:.*]] = tail call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
// CHECK-DXIL-NEXT: store target("dx.TypedBuffer", float, 1, 0, 0) %[[HANDLE]], ptr @Buf, align 4

// CHECK-NEXT: ret void
}
