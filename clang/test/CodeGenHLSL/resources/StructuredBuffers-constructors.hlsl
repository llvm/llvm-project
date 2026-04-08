// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:   llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:   llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,CHECK-SPV

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

// CHECK-DXIL: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", float, 0, 0) }
// CHECK-DXIL: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }
// CHECK-DXIL: %"class.hlsl::AppendStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }

// CHECK: @Buf1 = internal global %"class.hlsl::StructuredBuffer" poison
// CHECK: @[[Buf1Str:.*]] = private unnamed_addr constant [5 x i8] c"Buf1\00", align 1
// CHECK: @Buf2 = internal global %"class.hlsl::RWStructuredBuffer" poison
// CHECK: @[[Buf2Str:.*]] = private unnamed_addr constant [5 x i8] c"Buf2\00", align 1

// Buf1 initialization part 1 - global init function that calls StructuredBuffer<float>::__createFromBinding
// with explicit binding
// CHECK: define internal {{.*}}void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK: call void @hlsl::StructuredBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} @Buf1, i32 noundef 10, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[Buf1Str]])

// Buf1 initialization part 2 - body of StructuredBuffer<float>::::__createFromBinding

// CHECK: define {{.*}} void @hlsl::StructuredBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: ptr {{.*}} sret(%"class.hlsl::StructuredBuffer") align {{(4|8)}} %[[RetValue1:.*]], i32 noundef %registerNo, 
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name)
// CHECK: %[[Tmp1:.*]] = alloca %"class.hlsl::StructuredBuffer"
// CHECK-DXIL: %[[Handle1:.*]] = call target("dx.RawBuffer", float, 0, 0) 
// CHECK-DXIL-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0t(
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::StructuredBuffer", ptr %[[Tmp1]], i32 0, i32 0
// CHECK-DXIL: store target("dx.RawBuffer", float, 0, 0) %[[Handle1]], ptr %__handle, align 4
// CHECK: call void @hlsl::StructuredBuffer<float>::StructuredBuffer(hlsl::StructuredBuffer<float> const&)(ptr {{.*}} %[[RetValue1]], ptr {{.*}} %[[Tmp1]])

// Buf2 initialization part 1 - global init function that calls RWStructuredBuffer<float>::__createFromImplicitBindingWithImplicitCounter
// CHECK: define internal {{.*}}void @__cxx_global_var_init.1()
// CHECK-NEXT: entry:
// CHECK: call void @hlsl::RWStructuredBuffer<float>::__createFromImplicitBindingWithImplicitCounter(unsigned int, unsigned int, int, unsigned int, char const*, unsigned int)
// CHECK-SAME: (ptr {{.*}} @Buf2, i32 noundef 0, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[Buf2Str]], i32 noundef 1)

// Buf2 initialization part 2 - body of RWStructuredBuffer<float>::__createFromImplicitBindingWithImplicitCounter
// CHECK: define linkonce_odr hidden void @hlsl::RWStructuredBuffer<float>::__createFromImplicitBindingWithImplicitCounter(unsigned int, unsigned int, int, unsigned int, char const*, unsigned int)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWStructuredBuffer") align {{(4|8)}} %[[RetValue2:.*]], i32 noundef %orderId, 
// CHECK-SAME: i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name, i32 noundef %counterOrderId)
// CHECK: %[[Tmp2:.*]] = alloca %"class.hlsl::RWStructuredBuffer"
// CHECK-DXIL: %[[Handle2:.*]] = call target("dx.RawBuffer", float, 1, 0)
// CHECK-DXIL-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_f32_1_0t(
// CHECK-DXIL: %[[HandlePtr:.*]] = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %[[Tmp2]], i32 0, i32 0
// CHECK-DXIL-NEXT: store target("dx.RawBuffer", float, 1, 0) %[[Handle2]], ptr %[[HandlePtr]], align 4
// CHECK-DXIL: %[[HandlePtr:.*]] = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %[[Tmp2]], i32 0, i32 0
// CHECK-DXIL: %[[LoadedHandle:.*]] = load target("dx.RawBuffer", float, 1, 0), ptr %[[HandlePtr]], align 4
// CHECK-DXIL: %[[CounterHandlePtr:.*]] = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %[[Tmp2]], i32 0, i32 1
// CHECK-DXIL-NEXT: store target("dx.RawBuffer", float, 1, 0) %[[LoadedHandle]], ptr %[[CounterHandlePtr]], align 4
// CHECK-SPV: %[[Handle2:.*]] = call target("spirv.VulkanBuffer", [0 x float], 12, 1)
// CHECK-SPV-SAME: @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_a0f32_12_1t(
// CHECK-SPV: %[[HandlePtr:.*]] = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %[[Tmp2]], i32 0, i32 0
// CHECK-SPV-NEXT: store target("spirv.VulkanBuffer", [0 x float], 12, 1) %[[Handle2]], ptr %[[HandlePtr]], align 8
// CHECK-SPV: %[[HandlePtr:.*]] = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %[[Tmp2]], i32 0, i32 0
// CHECK-SPV: %[[LoadedHandle:.*]] = load target("spirv.VulkanBuffer", [0 x float], 12, 1), ptr %[[HandlePtr]], align 8
// CHECK-SPV: %[[CounterHandle:.*]] = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding
// CHECK-SPV: %[[CounterHandlePtr:.*]] = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %[[Tmp2]], i32 0, i32 1
// CHECK-SPV-NEXT: store target("spirv.VulkanBuffer", i32, 12, 1) %[[CounterHandle]], ptr %[[CounterHandlePtr]], align 8
// CHECK: call void @hlsl::RWStructuredBuffer<float>::RWStructuredBuffer(hlsl::RWStructuredBuffer<float> const&)(ptr {{.*}} %[[RetValue2]], ptr {{.*}} %[[Tmp2]])

// Buf3 initialization part 1 - local variable declared in function foo() is initialized by 
// AppendStructuredBuffer<float> C1 default constructor
// CHECK: define {{.*}}void @foo()
// CHECK-NEXT: entry:
// CHECK: %Buf3 = alloca %"class.hlsl::AppendStructuredBuffer", align {{4|8}}
// CHECK-NEXT: call void @hlsl::AppendStructuredBuffer<float>::AppendStructuredBuffer()(ptr {{.*}} %Buf3)

// Buf3 initialization part 2 - body of AppendStructuredBuffer<float> default C1 constructor that calls
// the default C2 constructor
// CHECK: define linkonce_odr hidden void @hlsl::StructuredBuffer<float>::StructuredBuffer()(ptr {{.*}} %this)
// CHECK: call void @hlsl::StructuredBuffer<float>::StructuredBuffer()(ptr {{.*}} %this1)

// Buf3 initialization part 3 - body of AppendStructuredBuffer<float> default C2 constructor that
// initializes handle to poison
// CHECK: define linkonce_odr hidden void @hlsl::StructuredBuffer<float>::StructuredBuffer()(ptr {{.*}} %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::AppendStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-DXIL: store target("dx.RawBuffer", float, 1, 0) poison, ptr %__handle, align 4

// Module initialization
// CHECK: define internal {{.*}}void @_GLOBAL__sub_I_StructuredBuffers_constructors.hlsl()
// CHECK-NEXT: entry:
// CHECK: call {{.*}}void @__cxx_global_var_init()
// CHECK-NEXT: call {{.*}}void @__cxx_global_var_init.1()
