// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented

StructuredBuffer<float> Buf : register(t10);
RWStructuredBuffer<float> Buf2 : register(u5, space1);

// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", float, 0, 0), float }
// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), float }

// CHECK: @Buf = global %"class.hlsl::StructuredBuffer" zeroinitializer, align 4
// CHECK: @Buf2 = global %"class.hlsl::RWStructuredBuffer" zeroinitializer, align 4

// CHECK: define linkonce_odr void @_ZN4hlsl16StructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this)
// CHECK-NEXT: entry:
// CHECK: define linkonce_odr void @_ZN4hlsl18RWStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this)
// CHECK-NEXT: entry:

// CHECK: define internal void @_GLOBAL__sub_I_StructuredBuffers_constructors.hlsl()
// CHECK: entry:
// CHECK: call void @_init_resource_bindings()

// CHECK: define internal void @_init_resource_bindings() {
// CHECK-NEXT: entry:
// CHECK-DXIL-NEXT: %Buf_h = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.handle.fromBinding.tdx.RawBuffer_f32_0_0t(i32 0, i32 10, i32 1, i32 0, i1 false)
// CHECK-DXIL-NEXT: store target("dx.RawBuffer", float, 0, 0) %Buf_h, ptr @Buf, align 4
// CHECK-DXIL-NEXT: %Buf2_h = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.handle.fromBinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 5, i32 1, i32 0, i1 false)
// CHECK-DXIL-NEXT: store target("dx.RawBuffer", float, 1, 0) %Buf2_h, ptr @Buf2, align 4
// CHECK-SPIRV-NEXT: %Buf_h = call target("dx.RawBuffer", float, 0, 0) @llvm.spv.handle.fromBinding.tdx.RawBuffer_f32_0_0t(i32 0, i32 10, i32 1, i32 0, i1 false)
// CHECK-SPIRV-NEXT: store target("dx.RawBuffer", float, 0, 0) %Buf_h, ptr @Buf", align 4
// CHECK-SPIRV-NEXT: %Buf2_h = call target("dx.RawBuffer", float, 1, 0) @llvm.spv.handle.fromBinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 5, i32 1, i32 0, i1 false)
// CHECK-SPIRV-NEXT: store target("dx.RawBuffer", float, 1, 0) %Buf2_h, ptr @Buf2", align 4
