// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | \
// RUN: llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// FIXME: SPIR-V codegen of llvm.spv.resource.handlefrombinding and resource types is not yet implemented
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | \
//        llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: Itanium ABI for C++ requires Clang to generate 2 constructors types to support polymorphism:
// - C1 - Complete object constructor - constructs the complete object, including virtual base classes.
// - C2 - Base object constructor - creates the object itself and initializes data members and non-virtual base classes.
// The constructors are distinquished by C1/C2 designators in their mangled name.
// https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-special-ctor-dtor

// Resource with explicit binding
RWBuffer<float> Buf1 : register(u5, space3);

// Resource with implicit binding
Buffer<double> Buf2;

export void foo() {
    // Local resource declaration
    RWBuffer<int> Buf3;
}

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
// CHECK: %"class.hlsl::Buffer" = type { target("dx.TypedBuffer", double, 0, 0, 0) }
// CHECK: %"class.hlsl::RWBuffer.0" = type { target("dx.TypedBuffer", i32, 1, 0, 1) }

// CHECK: @Buf1 = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @[[Buf1Str:.*]] = private unnamed_addr constant [5 x i8] c"Buf1\00", align 1
// CHECK: @Buf2 = internal global %"class.hlsl::Buffer" poison, align 4
// CHECK: @[[Buf2Str:.*]] = private unnamed_addr constant [5 x i8] c"Buf2\00", align 1

// Buf1 initialization part 1 - global init function that calls RWBuffer<float>::__createFromBinding
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 @Buf1, i32 noundef 5, i32 noundef 3, i32 noundef 1, i32 noundef 0, ptr noundef @[[Buf1Str]])

// Buf1 initialization part 2 - body of RWBuffer<float>::__createFromBinding
// CHECK: define {{.*}} void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[RetValue1:.*]], i32 noundef %registerNo, 
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK: %[[Tmp1:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[Handle1:.*]] = call target("dx.TypedBuffer", float, 1, 0, 0) 
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %[[Tmp1]], i32 0, i32 0
// CHECK-DXIL: store target("dx.TypedBuffer", float, 1, 0, 0) %[[Handle1]], ptr %__handle, align 4
// CHECK: call void @hlsl::RWBuffer<float>::RWBuffer(hlsl::RWBuffer<float> const&)(ptr {{.*}} %[[RetValue1]], ptr {{.*}} %[[Tmp1]])

// Buf2 initialization part 1 - global init function that RWBuffer<float>::__createFromImplicitBinding
// CHECK: define internal void @__cxx_global_var_init.1()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::Buffer<double>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} @Buf2, i32 noundef 0, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[Buf2Str]])

// Buf2 initialization part 2 - body of Buffer<double>::__createFromImplicitBinding call
// CHECK: define linkonce_odr hidden void @hlsl::Buffer<double>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::Buffer") align 4 %[[RetValue2:.*]], i32 noundef %orderId, 
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK: %[[Tmp2:.*]] = alloca %"class.hlsl::Buffer", align 4
// CHECK: %[[Handle2:.*]] = call target("dx.TypedBuffer", double, 0, 0, 0)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_f64_0_0_0t(
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::Buffer", ptr %[[Tmp2]], i32 0, i32 0
// CHECK-DXIL: store target("dx.TypedBuffer", double, 0, 0, 0) %[[Handle2]], ptr %__handle, align 4
// CHECK: call void @hlsl::Buffer<double>::Buffer(hlsl::Buffer<double> const&)(ptr {{.*}} %[[RetValue2]], ptr {{.*}} %[[Tmp2]])

// Buf3 initialization part 1 - local variable declared in function foo() is initialized by RWBuffer<int> C1 default constructor
// CHECK: define void @foo()
// CHECK-NEXT: entry:
// CHECK-NEXT: %Buf3 = alloca %"class.hlsl::RWBuffer.0", align 4
// CHECK-NEXT: call void @hlsl::RWBuffer<int>::RWBuffer()(ptr {{.*}} %Buf3)

// Buf3 initialization part 2 - body of RWBuffer<int> default C1 constructor that calls the default C2 constructor
// CHECK: define linkonce_odr hidden void @hlsl::RWBuffer<int>::RWBuffer()(ptr {{.*}} %this)
// CHECK: call void @hlsl::RWBuffer<int>::RWBuffer()(ptr {{.*}} %{{.*}})

// Buf3 initialization part 3 - body of RWBuffer<int> default C2 constructor that initializes handle to poison
// CHECK: define linkonce_odr hidden void @hlsl::RWBuffer<int>::RWBuffer()(ptr {{.*}} %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer.0", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: store target("dx.TypedBuffer", i32, 1, 0, 1) poison, ptr %__handle, align 4

// Module initialization
// CHECK: define internal void @_GLOBAL__sub_I_TypedBuffers_constructor.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__cxx_global_var_init()
// CHECK-NEXT: call void @__cxx_global_var_init.1()
