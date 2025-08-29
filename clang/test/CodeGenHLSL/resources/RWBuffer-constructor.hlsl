// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// FIXME: SPIR-V codegen of llvm.spv.resource.handlefrombinding and resource types is not yet implemented
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: Itanium ABI for C++ requires Clang to generate 2 constructors types to support polymorphism:
// - C1 - Complete object constructor - constructs the complete object, including virtual base classes.
// - C2 - Base object constructor - creates the object itself and initializes data members and non-virtual base classes.
// The constructors are distinquished by C1/C2 designators in their mangled name.
// https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-special-ctor-dtor

// Resource with explicit binding
RWBuffer<float> Buf1 : register(u5, space3);

// Resource with implicit binding
RWBuffer<double> Buf2;

export void foo() {
    // Local resource declaration
    RWBuffer<int> Buf3;
}

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
// CHECK: %"class.hlsl::RWBuffer.0" = type { target("dx.TypedBuffer", double, 1, 0, 0) }
// CHECK: %"class.hlsl::RWBuffer.1" = type { target("dx.TypedBuffer", i32, 1, 0, 1) }

// CHECK: @_ZL4Buf1 = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @[[Buf1Str:.*]] = private unnamed_addr constant [5 x i8] c"Buf1\00", align 1
// CHECK: @_ZL4Buf2 = internal global %"class.hlsl::RWBuffer.0" poison, align 4
// CHECK: @[[Buf2Str:.*]] = private unnamed_addr constant [5 x i8] c"Buf2\00", align 1

// Buf1 initialization part 1 - global init function that calls RWBuffer<float> C1 constructor with explicit binding
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf1,
// CHECK-SAME: i32 noundef 5, i32 noundef 3, i32 noundef 1, i32 noundef 0, ptr noundef @[[Buf1Str]])

// Buf1 initialization part 2 - body of RWBuffer<float> C1 constructor with explicit binding that calls the C2 constructor
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %registerNo, i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK: call void @_ZN4hlsl8RWBufferIfEC2EjjijPKc(ptr noundef nonnull align 4 dereferenceable(4)
// CHECK-SAME: %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr noundef %{{.*}})

// Buf2 initialization part 1 - global init function that calls RWBuffer<float> C1 constructor with implicit binding
// CHECK: define internal void @__cxx_global_var_init.1()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIdEC1EjijjPKc(ptr noundef nonnull align 4 dereferenceable(4) @_ZL4Buf2,
// CHECK-SAME: i32 noundef 0, i32 noundef 1, i32 noundef 0, i32 noundef 0, ptr noundef @[[Buf2Str]])

// Buf2 initialization part 2 - body of RWBuffer<float> C1 constructor with implicit binding that calls the C2 constructor
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIdEC1EjijjPKc(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, i32 noundef %orderId, ptr noundef %name)
// CHECK: call void @_ZN4hlsl8RWBufferIdEC2EjijjPKc(ptr noundef nonnull align 4 dereferenceable(4)
// CHECK-SAME: %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr noundef %{{.*}})

// Buf3 initialization part 1 - local variable declared in function foo() is initialized by RWBuffer<int> C1 default constructor
// CHECK: define void @_Z3foov()
// CHECK-NEXT: entry:
// CHECK-NEXT: %Buf3 = alloca %"class.hlsl::RWBuffer.1", align 4
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIiEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %Buf3)

// Buf3 initialization part 2 - body of RWBuffer<int> default C1 constructor that calls the default C2 constructor
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIiEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: call void @_ZN4hlsl8RWBufferIiEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %{{.*}})

// Buf1 initialization part 3 - body of RWBuffer<float> C2 constructor with explicit binding that initializes
// handle with @llvm.dx.resource.handlefrombinding 
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIfEC2EjjijPKc(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %registerNo, i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK-DXIL: %[[HANDLE:.*]] = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(
// CHECK-DXIL-SAME: i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-DXIL-NEXT: store target("dx.TypedBuffer", float, 1, 0, 0) %[[HANDLE]], ptr %__handle, align 4

// Buf2 initialization part 3 - body of RWBuffer<float> C2 constructor with implicit binding that initializes
// handle with @llvm.dx.resource.handlefromimplicitbinding
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIdEC2EjijjPKc(ptr noundef nonnull align 4 dereferenceable(4) %this,
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, i32 noundef %orderId, ptr noundef %name)
// CHECK: %[[HANDLE:.*]] = call target("dx.TypedBuffer", double, 1, 0, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_f64_1_0_0t
// CHECK-SAME: (i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer.0", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: store target("dx.TypedBuffer", double, 1, 0, 0) %[[HANDLE]], ptr %__handle, align 4

// Buf3 initialization part 3 - body of RWBuffer<int> default C2 constructor that initializes handle to poison
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIiEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer.1", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: store target("dx.TypedBuffer", i32, 1, 0, 1) poison, ptr %__handle, align 4

// Module initialization
// CHECK: define internal void @_GLOBAL__sub_I_RWBuffer_constructor.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__cxx_global_var_init()
// CHECK-NEXT: call void @__cxx_global_var_init.1()
