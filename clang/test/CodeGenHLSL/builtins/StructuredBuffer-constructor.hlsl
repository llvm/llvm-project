// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented
StructuredBuffer<float> Buf : register(u10);

// CHECK: define linkonce_odr noundef ptr @"??0?$StructuredBuffer@M@hlsl@@QAA@XZ"
// CHECK-NEXT: entry:

// CHECK: define internal void @_GLOBAL__sub_I_StructuredBuffer_constructor.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @"??__EBuf@@YAXXZ"()
// CHECK-NEXT: call void @_init_resource_bindings()

// CHECK: define internal void @_init_resource_bindings() {
// CHECK-NEXT: entry:
// CHECK-DXIL-NEXT: %Buf_h = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.handle.fromBinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 10, i32 1, i32 0, i1 false)
// CHECK-DXIL-NEXT: store target("dx.RawBuffer", float, 1, 0) %Buf_h, ptr @"?Buf@@3V?$StructuredBuffer@M@hlsl@@A", align 4
// CHECK-SPIRV-NEXT: %Buf_h = call target("dx.RawBuffer", float, 1, 0) @llvm.spv.handle.fromBinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 10, i32 1, i32 0, i1 false)
// CHECK-SPIRV-NEXT: store target("dx.RawBuffer", float, 1, 0) %Buf_h, ptr @"?Buf@@3V?$StructuredBuffer@M@hlsl@@A", align 4
