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

// Buf1 initialization part 1 - global init function that calls RWBuffer<float>::__createFromBinding
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc(ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 @_ZL4Buf1,
// CHECK-SAME: i32 noundef 5, i32 noundef 3, i32 noundef 1, i32 noundef 0, ptr noundef @[[Buf1Str]])

// Buf1 initialization part 2 - body of RWBuffer<float>::__createFromBinding
// CHECK: define {{.*}} void @_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc(
// CHECK-SAME: ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[Tmp1:.*]], i32 noundef %registerNo, 
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK: %[[Handle1:.*]] = call target("dx.TypedBuffer", float, 1, 0, 0) 
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(
// CHECK: call void @_ZN4hlsl8RWBufferIfEC1EU9_Res_u_CTfu17__hlsl_resource_t(
// CHECK-SAME: ptr {{.*}} %[[Tmp1]], target("dx.TypedBuffer", float, 1, 0, 0) %[[Handle1]])

// Buf2 initialization part 1 - global init function that RWBuffer<float>::__createFromImplicitBinding
// CHECK: define internal void @__cxx_global_var_init.1()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIdE27__createFromImplicitBindingEjjijPKc(ptr {{.*}} @_ZL4Buf2,
// CHECK-SAME: i32 noundef 0, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[Buf2Str]])

// Buf2 initialization part 2 - body of RWBuffer<float>::__createFromImplicitBinding call
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIdE27__createFromImplicitBindingEjjijPKc(
// CHECK-SAME: ptr {{.*}} sret(%"class.hlsl::RWBuffer.0") align 4 %[[Tmp2:.*]], i32 noundef %orderId, 
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK: %[[Handle2:.*]] = call target("dx.TypedBuffer", double, 1, 0, 0)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_f64_1_0_0t(
// CHECK: call void @_ZN4hlsl8RWBufferIdEC1EU9_Res_u_CTdu17__hlsl_resource_t(
// CHECK-SAME: ptr {{.*}} %[[Tmp2]], target("dx.TypedBuffer", double, 1, 0, 0) %[[Handle2]])

// Buf3 initialization part 1 - local variable declared in function foo() is initialized by RWBuffer<int> C1 default constructor
// CHECK: define void @_Z3foov()
// CHECK-NEXT: entry:
// CHECK-NEXT: %Buf3 = alloca %"class.hlsl::RWBuffer.1", align 4
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIiEC1Ev(ptr {{.*}} %Buf3)

// Buf3 initialization part 2 - body of RWBuffer<int> default C1 constructor that calls the default C2 constructor
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIiEC1Ev(ptr {{.*}} %this)
// CHECK: call void @_ZN4hlsl8RWBufferIiEC2Ev(ptr {{.*}} %{{.*}})

// Buf1 initialization part 3 - body of RWBuffer<float> constructor with handle
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIfEC2EU9_Res_u_CTfu17__hlsl_resource_t(ptr {{.*}} %this,
// CHECK-SAME: target("dx.TypedBuffer", float, 1, 0, 0) %handle)
// CHECK: %[[HandlePtr1:.*]] = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-DXIL-NEXT: store target("dx.TypedBuffer", float, 1, 0, 0) %{{.*}}, ptr %[[HandlePtr1]], align 4

// Buf2 initialization part 3 - body of RWBuffer<double> constructor with handle
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIdEC1EU9_Res_u_CTdu17__hlsl_resource_t(ptr {{.*}} %this,
// CHECK-SAME: target("dx.TypedBuffer", double, 1, 0, 0) %handle)
// CHECK: %[[HandlePtr2:.*]] = getelementptr inbounds nuw %"class.hlsl::RWBuffer.0", ptr %{{.*}}, i32 0, i32 0
// CHECK-DXIL-NEXT: store target("dx.TypedBuffer", double, 1, 0, 0) %{{.*}}, ptr %[[HandlePtr2]], align 4

// Buf3 initialization part 3 - body of RWBuffer<int> default C2 constructor that initializes handle to poison
// CHECK: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIiEC2Ev(ptr {{.*}} %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer.1", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: store target("dx.TypedBuffer", i32, 1, 0, 1) poison, ptr %__handle, align 4

// Module initialization
// CHECK: define internal void @_GLOBAL__sub_I_RWBuffer_constructor.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__cxx_global_var_init()
// CHECK-NEXT: call void @__cxx_global_var_init.1()
