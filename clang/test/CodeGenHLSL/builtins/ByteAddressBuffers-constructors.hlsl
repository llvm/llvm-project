// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource types is not yet implemented

ByteAddressBuffer Buf1: register(t1, space2);
RWByteAddressBuffer Buf2;

export void foo() {
  RasterizerOrderedByteAddressBuffer Buf3;
}

// CHECK: %"class.hlsl::ByteAddressBuffer" = type { target("dx.RawBuffer", i8, 0, 0) }
// CHECK: %"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }
// CHECK: %"class.hlsl::RasterizerOrderedByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 1) }

// CHECK: @_ZL4Buf1 = internal global %"class.hlsl::ByteAddressBuffer" poison, align 4
// CHECK: @_ZL4Buf2 = internal global %"class.hlsl::RWByteAddressBuffer" poison, align 4

//
// Buf1 initialization part 1 - constructor that creates handle from binding
//
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl17ByteAddressBufferC1Ejjij(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf1,
// CHECK-SAME: i32 noundef 2, i32 noundef 1, i32 noundef 1, i32 noundef 0)

// CHECK: define linkonce_odr void @_ZN4hlsl17ByteAddressBufferC1Ejjij(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %registerNo, i32 noundef %range, i32 noundef %index)
// CHECK: call void @_ZN4hlsl17ByteAddressBufferC2Ejjij(ptr noundef nonnull align 4 dereferenceable(4) %this1, 
// CHECK-SAME: i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3)

//
// Buf2 initialization part 1 - default constructor that initializes handle to poison
//

// CHECK: define internal void @__cxx_global_var_init.1()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl19RWByteAddressBufferC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf2)

// CHECK: define linkonce_odr void @_ZN4hlsl19RWByteAddressBufferC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: call void @_ZN4hlsl19RWByteAddressBufferC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this1)

//
// Buf3 initialization part 1 - local variable with default constructor that initializes handle to poison
//

// CHECK: define void @_Z3foov() #2 {
// CHECK-NEXT: entry:
// CHECK-NEXT: %Buf3 = alloca %"class.hlsl::RasterizerOrderedByteAddressBuffer", align 4
// CHECK-NEXT: call void @_ZN4hlsl34RasterizerOrderedByteAddressBufferC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %Buf3)

// CHECK: define linkonce_odr void @_ZN4hlsl34RasterizerOrderedByteAddressBufferC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: call void @_ZN4hlsl34RasterizerOrderedByteAddressBufferC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this1)

//
// Buf1 initialization part 2 - body of constructor that creates handle from binding
//
// CHECK: define linkonce_odr void @_ZN4hlsl17ByteAddressBufferC2Ejjij(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %registerNo, i32 noundef %range, i32 noundef %index)
// CHECK-DXIL: %[[HANDLE:.*]] = call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(
// CHECK-SAME: i32 %0, i32 %1, i32 %2, i32 %3, i1 false)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::ByteAddressBuffer", ptr %this1, i32 0, i32 0
// CHECK: store target("dx.RawBuffer", i8, 0, 0) %[[HANDLE]], ptr %[[HANDLE_PTR]], align 4

//
// Buf2 initialization part 2 - body of default constructor that initializes handle to poison
//
// CHECK: define linkonce_odr void @_ZN4hlsl19RWByteAddressBufferC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %this1, i32 0, i32 0
// CHECK: store i8 poison, ptr %[[HANDLE_PTR]], align 4

//
// Buf3 initialization part 2 - body of default constructor that initializes handle to poison
//
// CHECK: define linkonce_odr void @_ZN4hlsl34RasterizerOrderedByteAddressBufferC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RasterizerOrderedByteAddressBuffer", ptr %this1, i32 0, i32 0
// CHECK: store i8 poison, ptr %[[HANDLE_PTR]], align 4

//
// Module initialization
//

// CHECK: define internal void @_GLOBAL__sub_I_ByteAddressBuffers_constructors.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__cxx_global_var_init()
// CHECK-NEXT: call void @__cxx_global_var_init.1()
