// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// FIXME: SPIR-V codegen of llvm.spv.handle.fromBinding is not yet implemented
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented

RWBuffer<float> Buf : register(u5, space3);

// CHECK: define linkonce_odr noundef ptr @"??0?$RWBuffer@M@hlsl@@QAA@XZ"
// CHECK-NEXT: entry:

// CHECK: define internal void @_GLOBAL__sub_I_RWBuffer_constructor.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @"??__EBuf@@YAXXZ"()
// CHECK-NEXT: call void @_init_resource_bindings()

// CHECK: define internal void @_init_resource_bindings() {
// CHECK-NEXT: entry:
// CHECK-DXIL-NEXT: %Buf_h = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
// CHECK-DXIL-NEXT: store target("dx.TypedBuffer", float, 1, 0, 0) %Buf_h, ptr @"?Buf@@3V?$RWBuffer@M@hlsl@@A", align 4
// CHECK-SPIRV-NEXT: %Buf_h = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.spv.handle.fromBinding.tdx.TypedBuffer_f32_1_0_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
// CHECK-SPIRV-NEXT: store target("dx.TypedBuffer", float, 1, 0, 0) %Buf_h, ptr @"?Buf@@3V?$RWBuffer@M@hlsl@@A", align 4
