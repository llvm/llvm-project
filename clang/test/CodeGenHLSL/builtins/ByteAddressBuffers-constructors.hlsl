// RUN: %clang_cc1 -Wno-hlsl-implicit-binding -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// FIXME: SPIR-V codegen of llvm.spv.resource.handlefrombinding and resource types is not yet implemented
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: Itanium ABI for C++ requires Clang to generate 2 constructors types to support polymorphism:
// - C1 - Complete object constructor - constructs the complete object, including virtual base classes.
// - C2 - Base object constructor - creates the object itself and initializes data members and non-virtual base classes.
// The constructors are distinquished by C1/C2 designators in their mangled name.
// https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-special-ctor-dtor

// Resource with explicit binding
ByteAddressBuffer Buf1 : register(t1, space2);

// Resource with implicit binding
RWByteAddressBuffer Buf2;

export void foo() {
    // Local resource declaration
    RasterizerOrderedByteAddressBuffer Buf3;
}

// CHECK: %"class.hlsl::ByteAddressBuffer" = type { target("dx.RawBuffer", i8, 0, 0) }
// CHECK: %"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }
// CHECK: %"class.hlsl::RasterizerOrderedByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 1) }

// CHECK: @_ZL4Buf1 = internal global %"class.hlsl::ByteAddressBuffer" poison, align 4
// CHECK: @_ZL4Buf2 = internal global %"class.hlsl::RWByteAddressBuffer" poison, align 4

// Buf1 initialization part 1 - global init function that calls ByteAddressBuffer C1 constructor with explicit binding
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl17ByteAddressBufferC1Ejjij(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf1,
// CHECK-SAME: i32 noundef 1, i32 noundef 2, i32 noundef 1, i32 noundef 0)

// Buf1 initialization part 2 - body of ByteAddressBuffer C1 constructor with explicit binding that calls the C2 constructor
// CHECK: define linkonce_odr void @_ZN4hlsl17ByteAddressBufferC1Ejjij(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %registerNo, i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index)
// CHECK-NEXT: entry:
// CHECK-NEXT: %this.addr = alloca ptr, align 4
// CHECK-NEXT: %registerNo.addr = alloca i32, align 4
// CHECK-NEXT: %spaceNo.addr = alloca i32, align 4
// CHECK-NEXT: %range.addr = alloca i32, align 4
// CHECK-NEXT: %index.addr = alloca i32, align 4
// CHECK-NEXT: store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT: store i32 %registerNo, ptr %registerNo.addr, align 4
// CHECK-NEXT: store i32 %spaceNo, ptr %spaceNo.addr, align 4
// CHECK-NEXT: store i32 %range, ptr %range.addr, align 4
// CHECK-NEXT: store i32 %index, ptr %index.addr, align 4
// CHECK-NEXT: %this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT: %0 = load i32, ptr %registerNo.addr, align 4
// CHECK-NEXT: %1 = load i32, ptr %spaceNo.addr, align 4
// CHECK-NEXT: %2 = load i32, ptr %range.addr, align 4
// CHECK-NEXT: %3 = load i32, ptr %index.addr, align 4
// CHECK: call void @_ZN4hlsl17ByteAddressBufferC2Ejjij(ptr noundef nonnull align 4 dereferenceable(4) %this1, 
// CHECK-SAME: i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3)
// CHECK-NEXT: ret void

// Buf2 initialization part 1 - FIXME: constructor with implicit binding does not exist yet; 
// the global init function currently calls the default RWByteAddressBuffer C1 constructor
// CHECK: define internal void @__cxx_global_var_init.1()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl19RWByteAddressBufferC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf2)

// Buf3 initialization part 1 - local variable declared in function foo() is initialized by 
// RasterizerOrderedByteAddressBuffer C1 default constructor
// CHECK: define void @_Z3foov() #2 {
// CHECK-NEXT: entry:
// CHECK-NEXT: %Buf3 = alloca %"class.hlsl::RasterizerOrderedByteAddressBuffer", align 4
// CHECK-NEXT: call void @_ZN4hlsl34RasterizerOrderedByteAddressBufferC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %Buf3)

// Buf3 initialization part 2 - body of RasterizerOrderedByteAddressBuffer default C1 constructor that
// calls the default C2 constructor
// CHECK: define linkonce_odr void @_ZN4hlsl34RasterizerOrderedByteAddressBufferC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK-NEXT: entry:
// CHECK-NEXT: %this.addr = alloca ptr, align 4
// CHECK-NEXT: store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT: %this1 = load ptr, ptr %this.addr, align 4
// CHECK: call void @_ZN4hlsl34RasterizerOrderedByteAddressBufferC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this1)
// CHECK-NEXT: ret void

// Buf1 initialization part 3 - ByteAddressBuffer C2 constructor with explicit binding that initializes
// handle with @llvm.dx.resource.handlefrombinding
// CHECK: define linkonce_odr void @_ZN4hlsl17ByteAddressBufferC2Ejjij(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %registerNo, i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index)
// CHECK-NEXT: entry:
// CHECK-NEXT: %this.addr = alloca ptr, align 4
// CHECK-NEXT: %registerNo.addr = alloca i32, align 4
// CHECK-NEXT: %spaceNo.addr = alloca i32, align 4
// CHECK-NEXT: %range.addr = alloca i32, align 4
// CHECK-NEXT: %index.addr = alloca i32, align 4
// CHECK-NEXT: store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT: store i32 %registerNo, ptr %registerNo.addr, align 4
// CHECK-NEXT: store i32 %spaceNo, ptr %spaceNo.addr, align 4
// CHECK-NEXT: store i32 %range, ptr %range.addr, align 4
// CHECK-NEXT: store i32 %index, ptr %index.addr, align 4
// CHECK-NEXT: %this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT: %0 = load i32, ptr %registerNo.addr, align 4
// CHECK-NEXT: %1 = load i32, ptr %spaceNo.addr, align 4
// CHECK-NEXT: %2 = load i32, ptr %range.addr, align 4
// CHECK-NEXT: %3 = load i32, ptr %index.addr, align 4
// CHECK-DXIL-NEXT: %4 = call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(
// CHECK-DXIL-SAME: i32 %1, i32 %0, i32 %2, i32 %3, i1 false)
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::ByteAddressBuffer", ptr %this1, i32 0, i32 0
// CHECK-DXIL-NEXT: store target("dx.RawBuffer", i8, 0, 0) %4, ptr %__handle, align 4
// CHECK-NEXT: ret void

// Buf3 initialization part 3 - body of RasterizerOrderedByteAddressBuffer default C2 constructor that
// initializes handle to poison
// CHECK: define linkonce_odr void @_ZN4hlsl34RasterizerOrderedByteAddressBufferC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RasterizerOrderedByteAddressBuffer", ptr %this1, i32 0, i32 0
// CHECK: store target("dx.RawBuffer", i8, 1, 1) poison, ptr %__handle, align 4

// Module initialization
// CHECK: define internal void @_GLOBAL__sub_I_ByteAddressBuffers_constructors.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__cxx_global_var_init()
// CHECK-NEXT: call void @__cxx_global_var_init.1()
