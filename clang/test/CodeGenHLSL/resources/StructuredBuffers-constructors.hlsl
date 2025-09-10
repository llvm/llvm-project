// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// FIXME: SPIR-V codegen of llvm.spv.resource.handlefrombinding and resource types is not yet implemented
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: Itanium ABI for C++ requires Clang to generate 2 constructors types to support polymorphism:
// - C1 - Complete object constructor - constructs the complete object, including virtual base classes.
// - C2 - Base object constructor - creates the object itself and initializes data members and non-virtual base classes.
// The constructors are distinquished by C1/C2 designators in their mangled name.
// https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-special-ctor-dtor

// Resource with explicit binding
StructuredBuffer<float> Buf1 : register(t10, space2);

// Resource with implicit binding
RWStructuredBuffer<float> Buf2;

export void foo() {
  AppendStructuredBuffer<float> Buf3;
}

// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", float, 0, 0) }
// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }
// CHECK: %"class.hlsl::AppendStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }

// CHECK: @_ZL4Buf1 = internal global %"class.hlsl::StructuredBuffer" poison, align 4
// CHECK: @[[Buf1Str:.*]] = private unnamed_addr constant [5 x i8] c"Buf1\00", align 1
// CHECK: @_ZL4Buf2 = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4
// CHECK: @[[Buf2Str:.*]] = private unnamed_addr constant [5 x i8] c"Buf2\00", align 1

// Buf1 initialization part 1 - global init function that calls StructuredBuffer<float> C1 constructor
// with explicit binding
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl16StructuredBufferIfEC1EjjijPKc(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf1,
// CHECK-SAME: i32 noundef 10, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[Buf1Str]])

// Buf1 initialization part 2 - body of StructuredBuffer<float> C1 constructor with explicit binding 
// that calls the C2 constructor
// CHECK: define linkonce_odr hidden void @_ZN4hlsl16StructuredBufferIfEC1EjjijPKc(ptr noundef nonnull align 4 dereferenceable(4) %this, 
// CHECK-SAME: i32 noundef %registerNo, i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK: call void @_ZN4hlsl16StructuredBufferIfEC2EjjijPKc(ptr noundef nonnull align 4 dereferenceable(4)
// CHECK-SAME: %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr noundef %{{.*}})

// Buf2 initialization part 1 - global init function that calls RWStructuredBuffer<float> C1 constructor with
// implicit binding
// CHECK: define internal void @__cxx_global_var_init.1()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl18RWStructuredBufferIfEC1EjijjPKc(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf2,
// CHECK-SAME: i32 noundef 0, i32 noundef 1, i32 noundef 0, i32 noundef 0, ptr noundef @[[Buf2Str]])

// Buf2 initialization part 2 - body of RWStructuredBuffer<float> C1 constructor with implicit binding that calls the C2 constructor
// CHECK: define linkonce_odr hidden void @_ZN4hlsl18RWStructuredBufferIfEC1EjijjPKc(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, i32 noundef %orderId, ptr noundef %name)
// CHECK: call void @_ZN4hlsl18RWStructuredBufferIfEC2EjijjPKc(ptr noundef nonnull align 4 dereferenceable(4)
// CHECK-SAME: %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr noundef %{{.*}})

// Buf3 initialization part 1 - local variable declared in function foo() is initialized by 
// AppendStructuredBuffer<float> C1 default constructor
// CHECK: define void @_Z3foov()
// CHECK-NEXT: entry:
// CHECK-NEXT: %Buf3 = alloca %"class.hlsl::AppendStructuredBuffer", align 4
// CHECK-NEXT: call void @_ZN4hlsl22AppendStructuredBufferIfEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %Buf3)

// Buf3 initialization part 2 - body of AppendStructuredBuffer<float> default C1 constructor that calls
// the default C2 constructor
// CHECK: define linkonce_odr hidden void @_ZN4hlsl22AppendStructuredBufferIfEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: call void @_ZN4hlsl22AppendStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %{{.*}})

// Buf1 initialization part 3 - body of AppendStructuredBuffer<float> C2 constructor with explicit binding 
// that initializes handle with @llvm.dx.resource.handlefrombinding
// CHECK: define linkonce_odr hidden void @_ZN4hlsl16StructuredBufferIfEC2EjjijPKc(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %registerNo, i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK-DXIL: %[[HANDLE:.*]] = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0t(
// CHECK-SAME: i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::StructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-DXIL-NEXT: store target("dx.RawBuffer", float, 0, 0) %[[HANDLE]], ptr %__handle, align 4

// Buf2 initialization part 3 - body of RWStructuredBuffer<float> C2 constructor with implicit binding that initializes
// handle with @llvm.dx.resource.handlefromimplicitbinding
// CHECK: define linkonce_odr hidden void @_ZN4hlsl18RWStructuredBufferIfEC2EjijjPKc(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, i32 noundef %orderId, ptr noundef %name)
// CHECK: %[[HANDLE:.*]] = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_f32_1_0t
// CHECK-SAME: (i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: store target("dx.RawBuffer", float, 1, 0) %[[HANDLE]], ptr %__handle, align 4

// Buf3 initialization part 3 - body of AppendStructuredBuffer<float> default C2 constructor that
// initializes handle to poison
// CHECK: define linkonce_odr hidden void @_ZN4hlsl22AppendStructuredBufferIfEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::AppendStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK: store target("dx.RawBuffer", float, 1, 0) poison, ptr %__handle, align 4

// Module initialization
// CHECK: define internal void @_GLOBAL__sub_I_StructuredBuffers_constructors.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__cxx_global_var_init()
// CHECK-NEXT: call void @__cxx_global_var_init.1()
