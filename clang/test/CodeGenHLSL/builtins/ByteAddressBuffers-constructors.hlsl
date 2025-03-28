// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented

ByteAddressBuffer Buffer0: register(t0);
RWByteAddressBuffer Buffer1: register(u1, space2);
RasterizerOrderedByteAddressBuffer Buffer2: register(u3, space4);

// CHECK: "class.hlsl::ByteAddressBuffer" = type { target("dx.RawBuffer", i8, 0, 0) }
// CHECK: "class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }
// CHECK: "class.hlsl::RasterizerOrderedByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 1) }

// CHECK: @_ZL7Buffer0 = internal global %"class.hlsl::ByteAddressBuffer" poison, align 4
// CHECK: @_ZL7Buffer1 = internal global %"class.hlsl::RWByteAddressBuffer" poison, align 4
// CHECK: @_ZL7Buffer2 = internal global %"class.hlsl::RasterizerOrderedByteAddressBuffer" poison, align 4

// CHECK; define internal void @_init_resource_Buffer0()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", i8, 0, 0) [[H]], ptr @_ZL7Buffer0, align 4

// CHECK; define internal void @_init_resource_Buffer1()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 2, i32 1, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", i8, 1, 0) [[H]], ptr @_ZL7Buffer1, align 4

// CHECK; define internal void @_init_resource_Buffer2()
// CHECK-DXIL: [[H:%.*]] = call target("dx.RawBuffer", i8, 1, 1) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_1t(i32 4, i32 3, i32 1, i32 0, i1 false)
// CHECK-DXIL: store target("dx.RawBuffer", i8, 1, 1) [[H]], ptr @_ZL7Buffer2, align 4

// CHECK: define internal void @_GLOBAL__sub_I_ByteAddressBuffers_constructors.hlsl()
// CHECK: entry:
// CHECK: call void @_init_resource__ZL7Buffer0()
// CHECK: call void @_init_resource__ZL7Buffer1()
// CHECK: call void @_init_resource__ZL7Buffer2()
