// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -DSPIRV -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV


StructuredBuffer<float> Buf : register(t10);
RWStructuredBuffer<float> Buf2 : register(u5, space1);

#ifndef SPIRV
// NOTE: SPIRV codegen for these resource types is not implemented yet.
AppendStructuredBuffer<float> Buf3 : register(u3);
ConsumeStructuredBuffer<float> Buf4 : register(u4);
RasterizerOrderedStructuredBuffer<float> Buf5 : register(u1, space2);
#endif

// CHECK-DXIL: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", float, 0, 0) }
// CHECK-DXIL: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }
// CHECK-DXIL: %"class.hlsl::AppendStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }
// CHECK-DXIL: %"class.hlsl::ConsumeStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }
// CHECK-DXIL: %"class.hlsl::RasterizerOrderedStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 1) }

// CHECK-SPIRV: %"class.hlsl::StructuredBuffer" = type { target("spirv.VulkanBuffer", [0 x float], 12, 0) }
// CHECK-SPIRV: %"class.hlsl::RWStructuredBuffer" = type { target("spirv.VulkanBuffer", [0 x float], 12, 1) }


// CHECK: @_ZL3Buf = internal global %"class.hlsl::StructuredBuffer" poison
// CHECK: @_ZL4Buf2 = internal global %"class.hlsl::RWStructuredBuffer" poison
// CHECK-DXIL: @_ZL4Buf3 = internal global %"class.hlsl::AppendStructuredBuffer" poison, align 4
// CHECK-DXIL: @_ZL4Buf4 = internal global %"class.hlsl::ConsumeStructuredBuffer" poison, align 4
// CHECK-DXIL: @_ZL4Buf5 = internal global %"class.hlsl::RasterizerOrderedStructuredBuffer" poison, align 4

// CHECK: define internal void @_init_resource__ZL3Buf()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0t(i32 0, i32 10, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 0, 0) [[H]], ptr @_ZL3Buf, align 4
// CHECK-SPIRV: [[H:%.*]] = call target("spirv.VulkanBuffer", [0 x float], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_0t(i32 0, i32 10, i32 1, i32 0, i1 false)
// CHECK-SPIRV: store target("spirv.VulkanBuffer", [0 x float], 12, 0) [[H]], ptr @_ZL3Buf, align 8

// CHECK: define internal void @_init_resource__ZL4Buf2()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 5, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 0) [[H]], ptr @_ZL4Buf2, align 4
// CHECK-SPIRV: [[H:%.*]] = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 1, i32 5, i32 1, i32 0, i1 false)
// CHECK-SPIRV: store target("spirv.VulkanBuffer", [0 x float], 12, 1) [[H]], ptr @_ZL4Buf2, align 8

// CHECK-DXIL: define internal void @_init_resource__ZL4Buf3()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 3, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 0) [[H]], ptr @_ZL4Buf3, align 4

// CHECK-DXIL: define internal void @_init_resource__ZL4Buf4()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 4, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 0) [[H]], ptr @_ZL4Buf4, align 4

// CHECK-DXIL: define internal void @_init_resource__ZL4Buf5()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 1, 1) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_1t(i32 2, i32 1, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 1) [[H]], ptr @_ZL4Buf5, align 4

// CHECK: define linkonce_odr void @_ZN4hlsl16StructuredBufferIfEC2Ev(ptr noundef nonnull align {{[48]}} dereferenceable({{[48]}}) %this)
// CHECK-NEXT: entry:
// CHECK-DXIL: define linkonce_odr void @_ZN4hlsl18RWStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-DXIL-NEXT: entry:
// CHECK-DXIL: define linkonce_odr void @_ZN4hlsl22AppendStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-DXIL-NEXT: entry:
// CHECK-DXIL: define linkonce_odr void @_ZN4hlsl23ConsumeStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-DXIL: define linkonce_odr void @_ZN4hlsl33RasterizerOrderedStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-DXIL-NEXT: entry:

// CHECK: define {{.*}} void @_GLOBAL__sub_I_StructuredBuffers_constructors.hlsl()
// CHECK: call {{.*}} @_init_resource__ZL3Buf()
// CHECK: call {{.*}} @_init_resource__ZL4Buf2()
// CHECK-DXIL: call void @_init_resource__ZL4Buf3()
// CHECK-DXIL: call void @_init_resource__ZL4Buf4()
// CHECK-DXIL: call void @_init_resource__ZL4Buf5()
