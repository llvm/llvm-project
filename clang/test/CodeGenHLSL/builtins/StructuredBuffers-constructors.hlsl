// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented

StructuredBuffer<float> Buf : register(t10);
RWStructuredBuffer<float> Buf2 : register(u5, space1);
AppendStructuredBuffer<float> Buf3 : register(u3);
ConsumeStructuredBuffer<float> Buf4 : register(u4);
RasterizerOrderedStructuredBuffer<float> Buf5 : register(u1, space2);

// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", float, 0, 0) }
// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }
// CHECK: %"class.hlsl::AppendStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }
// CHECK: %"class.hlsl::ConsumeStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }
// CHECK: %"class.hlsl::RasterizerOrderedStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 1) }

// CHECK: @_ZL3Buf = internal global %"class.hlsl::StructuredBuffer" poison, align 4
// CHECK: @_ZL4Buf2 = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4
// CHECK: @_ZL4Buf3 = internal global %"class.hlsl::AppendStructuredBuffer" poison, align 4
// CHECK: @_ZL4Buf4 = internal global %"class.hlsl::ConsumeStructuredBuffer" poison, align 4
// CHECK: @_ZL4Buf5 = internal global %"class.hlsl::RasterizerOrderedStructuredBuffer" poison, align 4

// CHECK: define internal void @_init_resource__ZL3Buf()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0t(i32 0, i32 10, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 0, 0) [[H]], ptr @_ZL3Buf, align 4

// CHECK: define internal void @_init_resource__ZL4Buf2()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 5, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 0) [[H]], ptr @_ZL4Buf2, align 4

// CHECK: define internal void @_init_resource__ZL4Buf3()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 3, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 0) [[H]], ptr @_ZL4Buf3, align 4

// CHECK: define internal void @_init_resource__ZL4Buf4()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 4, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 0) [[H]], ptr @_ZL4Buf4, align 4

// CHECK: define internal void @_init_resource__ZL4Buf5()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", float, 1, 1) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_1t(i32 2, i32 1, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 1) [[H]], ptr @_ZL4Buf5, align 4

// CHECK: define linkonce_odr void @_ZN4hlsl16StructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-NEXT: entry:
// CHECK: define linkonce_odr void @_ZN4hlsl18RWStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-NEXT: entry:
// CHECK: define linkonce_odr void @_ZN4hlsl22AppendStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-NEXT: entry:
// CHECK: define linkonce_odr void @_ZN4hlsl23ConsumeStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: define linkonce_odr void @_ZN4hlsl33RasterizerOrderedStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-NEXT: entry:

// CHECK: define internal void @_GLOBAL__sub_I_StructuredBuffers_constructors.hlsl()
// CHECK: call void @_init_resource__ZL3Buf()
// CHECK: call void @_init_resource__ZL4Buf2()
// CHECK: call void @_init_resource__ZL4Buf3()
// CHECK: call void @_init_resource__ZL4Buf4()
// CHECK: call void @_init_resource__ZL4Buf5()
