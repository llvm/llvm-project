// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// FIXME: SPIR-V codegen of llvm.spv.handle.fromBinding is not yet implemented
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented

RWBuffer<float> Buf : register(u5, space3);

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
// CHECK: @Buf = global %"class.hlsl::RWBuffer" zeroinitializer, align 4

// CHECK: define void @main()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_GLOBAL__sub_I_RWBuffer_constructor.hlsl()
// CHECK-NEXT: call void @_init_resource_bindings()
// CHECK-NEXT: call void @_Z4mainv()
// CHECK-NEXT: ret void

// CHECK: define linkonce_odr void @_ZN4hlsl8RWBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-NEXT: entry:

// CHECK: define internal void @_init_resource_bindings() {
// CHECK-NEXT: entry:
// CHECK-DXIL-NEXT: %Buf_h = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
// CHECK-DXIL-NEXT: store target("dx.TypedBuffer", float, 1, 0, 0) %Buf_h, ptr @Buf, align 4
// CHECK-SPIRV-NEXT: %Buf_h = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.spv.handle.fromBinding.tdx.TypedBuffer_f32_1_0_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
// CHECK-SPIRV-NEXT: store target("dx.TypedBuffer", float, 1, 0, 0) %Buf_h, ptr @Buf, align 4

[numthreads(4,1,1)]
void main() {}
