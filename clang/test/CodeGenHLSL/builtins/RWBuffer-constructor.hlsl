// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// FIXME: SPIR-V codegen of llvm.spv.resource.handlefrombinding is not yet implemented
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented

RWBuffer<float> Buf1 : register(u5, space3);
RWBuffer<double> Buf2;

export void foo() {
    RWBuffer<int> Buf3;
}

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
// CHECK: %"class.hlsl::RWBuffer.0" = type { target("dx.TypedBuffer", double, 1, 0, 0) }
// CHECK: %"class.hlsl::RWBuffer.1" = type { target("dx.TypedBuffer", i32, 1, 0, 1) }

// CHECK: @_ZL4Buf1 = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @_ZL4Buf2 = internal global %"class.hlsl::RWBuffer.0" poison, align 4

//
// Buf1 initialization part 1 - constructor that creates handle from binding
//
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1Ejjij(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf1,
// CHECK-SAME: i32 noundef 3, i32 noundef 5, i32 noundef 1, i32 noundef 0)

// CHECK: define linkonce_odr void @_ZN4hlsl8RWBufferIfEC1Ejjij(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %registerNo, i32 noundef %range, i32 noundef %index)
// CHECK: call void @_ZN4hlsl8RWBufferIfEC2Ejjij(

//
// Buf2 initialization part 1 - default constructor that initializes handle to poison
//

// CHECK: define internal void @__cxx_global_var_init.1() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIdEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf2)

// CHECK: define linkonce_odr void @_ZN4hlsl8RWBufferIdEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: call void @_ZN4hlsl8RWBufferIdEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this1)

//
// Buf3 initialization part 1 - local variable with default constructor that initializes handle to poison
//
// CHECK: define void @_Z3foov()
// CHECK-NEXT: entry:
// CHECK-NEXT: %Buf3 = alloca %"class.hlsl::RWBuffer.1", align 4
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIiEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %Buf3)

// CHECK: define linkonce_odr void @_ZN4hlsl8RWBufferIiEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: call void @_ZN4hlsl8RWBufferIiEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this1)

//
// Buf1 initialization part 2 - body of constructor that creates handle from binding
//
// CHECK: define linkonce_odr void @_ZN4hlsl8RWBufferIfEC2Ejjij(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %registerNo, i32 noundef %range, i32 noundef %index)
// CHECK-DXIL: %[[HANDLE:.*]] = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %this1, i32 0, i32 0
// CHECK: store target("dx.TypedBuffer", float, 1, 0, 0) %[[HANDLE]], ptr %[[HANDLE_PTR]], align 4

//
// Buf2 initialization part 2 - body of default constructor that initializes handle to poison
//
// CHECK: define linkonce_odr void @_ZN4hlsl8RWBufferIdEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWBuffer.0", ptr %this1, i32 0, i32 0
// CHECK: store target("dx.TypedBuffer", double, 1, 0, 0) poison, ptr %[[HANDLE_PTR]], align 4

//
// Buf3 initialization part 2 - body of default constructor that initializes handle to poison
//
// CHECK: define linkonce_odr void @_ZN4hlsl8RWBufferIiEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWBuffer.1", ptr %this1, i32 0, i32 0
// CHECK: store target("dx.TypedBuffer", i32, 1, 0, 1) poison, ptr %[[HANDLE_PTR]], align 4

//
// Module initialization
//
// CHECK: define internal void @_GLOBAL__sub_I_RWBuffer_constructor.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__cxx_global_var_init()
// CHECK-NEXT: call void @__cxx_global_var_init.1()
