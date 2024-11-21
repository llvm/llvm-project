// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented

ByteAddressBuffer Buffer: register(t0);

// CHECK: class.hlsl::ByteAddressBuffer" = type <{ target("dx.RawBuffer", i8, 1, 1), i8, [3 x i8] }>

// CHECK: @Buffer = global %"class.hlsl::ByteAddressBuffer" zeroinitializer, align 4

// CHECK: define internal void @_GLOBAL__sub_I_ByteAddressBuffer_contructor.hlsl()
// CHECK: entry:
// CHECK: call void @_init_resource_bindings()

// CHECK: define internal void @_init_resource_bindings() {
// CHECK-NEXT: entry:
// CHECK-DXIL-NEXT: %Buffer_h = call target("dx.RawBuffer", i8, 1, 1) @llvm.dx.handle.fromBinding.tdx.RawBuffer_i8_1_1t(i32 0, i32 0, i32 1, i32 0, i1 false)
// CHECK-DXIL-NEXT: store target("dx.RawBuffer", i8, 1, 1) %Buffer_h, ptr @Buffer, align 4
