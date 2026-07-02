// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:        llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL -DCONST_ADDR_SPACE=2 -DPADDING_TYPE="dx.Padding"
// RUN: %clang_cc1 -triple spirv-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:        llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,CHECK-SPV -DCONST_ADDR_SPACE=12 -DPADDING_TYPE="spirv.Padding"

struct S {
    float3 f3;
    int a;
};

struct MyConstants {
    float f;
    int2 i2;
    half3 h3;
    double d;
    int array[2];
    float2x2 m;
    S s;
};

ConstantBuffer<MyConstants> CB;
ConstantBuffer<S> CBArray[2];

// CHECK-DXIL: %"class.hlsl::ConstantBuffer" = type { target("dx.CBuffer", %MyConstants) }
// CHECK-SPV: %"class.hlsl::ConstantBuffer" = type { target("spirv.VulkanBuffer", %MyConstants, 2, 0) }

// CHECK: %MyConstants = type <{ float, <2 x i32>, target("[[PADDING_TYPE]]", 4), <3 x float>,
// CHECK-SAME: target("[[PADDING_TYPE]]", 4), double, target("[[PADDING_TYPE]]", 8),
// CHECK-SAME: <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }>,
// CHECK-SAME: target("[[PADDING_TYPE]]", 12), <{ [1 x <{ <2 x float>, target("[[PADDING_TYPE]]", 8) }>],
// CHECK-SAME: <2 x float> }>, target("[[PADDING_TYPE]]", 8), %S }>

// CHECK: %S = type <{ <3 x float>, i32 }>
// CHECK: %struct.S = type { <3 x float>, i32 }

// CHECK: @CB = internal global %"class.hlsl::ConstantBuffer" poison, align {{(4|8)}}
// CHECK: [[CBStr:.*]] = private unnamed_addr constant [3 x i8] c"CB\00", align 1
// CHECK: [[CBArrayStr:.*]] = private unnamed_addr constant [8 x i8] c"CBArray\00", align 1

// CB initialization
//
// CHECK-LABEL: __cxx_global_var_init
// CHECK: call void @hlsl::ConstantBuffer<MyConstants>::__createFromImplicitBinding({{[^)]+}})
// CHECK-SAME: (ptr dead_on_unwind writable sret(%"class.hlsl::ConstantBuffer") align {{(4|8)}} @CB,
// CHECK-SAME: i32 noundef 0, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef [[CBStr]])

// CHECK: define linkonce_odr hidden void @hlsl::ConstantBuffer<MyConstants>::__createFromImplicitBinding(
// CHECK-DXIL: call target("dx.CBuffer", %MyConstants) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s_MyConstantsst
// CHECK-SPV: call target("spirv.VulkanBuffer", %MyConstants, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s_MyConstantss_2_0t(

// CHECK-LABEL: TestElementAccess
void TestElementAccess() {
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CB)
// CHECK-NEXT: [[CB_F_PTR:%.*]] = getelementptr inbounds nuw %MyConstants, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CB_F:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_F_PTR]], align 4
// CHECK-NEXT: store float [[CB_F]], ptr %f, align 4
    float f = CB.f;

// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CB)
// CHECK-NEXT: [[CB_I2_PTR:%.*]] = getelementptr inbounds nuw %MyConstants, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 1
// CHECK-NEXT: [[CB_I2:%.*]] = load <2 x i32>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_I2_PTR]], align 4
// CHECK-NEXT: store <2 x i32> [[CB_I2]], ptr %i2, align 4
    int2 i2 = CB.i2;

// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CB)
// CHECK-NEXT: [[CB_H3_PTR:%.*]] = getelementptr inbounds nuw %MyConstants, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 3
// CHECK-NEXT: [[CB_H3:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_H3_PTR]], align 4
// CHECK-NEXT: store <3 x float> [[CB_H3]], ptr %h3, align 4
    half3 h3 = CB.h3;

// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CB)
// CHECK-NEXT: [[CB_D_PTR:%.*]] = getelementptr inbounds nuw %MyConstants, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 5
// CHECK-NEXT: [[CB_D:%.*]] = load double, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_D_PTR]], align 8
// CHECK-NEXT: store double [[CB_D]], ptr %d, align 8
    double d = CB.d;
    
// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CB)
// CHECK-NEXT: [[CB_ARRAY_PTR:%.*]] = getelementptr inbounds nuw %MyConstants, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 7
// CHECK-NEXT: [[CB_ARRAY_DECAY_PTR:%.*]] = getelementptr inbounds [2 x i32], ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARRAY_PTR]], {{(i32|i64)}} 0, {{(i32|i64)}} 0
// CHECK-NEXT: [[CB_ARRAY_1_PTR:%.*]] = getelementptr <{ i32, target("[[PADDING_TYPE]]", 12) }>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARRAY_DECAY_PTR]], {{(i32|i64)}} 1, {{(i32|i64)}} 0
// CHECK-NEXT: [[CB_ARRAY_1:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARRAY_1_PTR]], align 16
// CHECK-NEXT: store i32 [[CB_ARRAY_1]], ptr %arrayEl, align 4
    int arrayEl = CB.array[1];

// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CB)
// CHECK-NEXT: [[CB_M_PTR:%.*]] = getelementptr inbounds nuw %MyConstants, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 9
// CHECK-NEXT: [[CB_M:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_M_PTR]], align 4
// CHECK-NEXT: store <4 x float> [[CB_M]], ptr %m, align 4
    float2x2 m = CB.m;

// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CB)
// CHECK-NEXT: [[CB_S_PTR:%.*]] = getelementptr inbounds nuw %MyConstants, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 11
// CHECK-NEXT: [[CB_S_F3_PTR:%.*]] = getelementptr inbounds %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_S_PTR]], i32 0, i32 0
// CHECK-NEXT: [[S_F3_PTR:%.*]] = getelementptr inbounds %struct.S, ptr %s, i32 0, i32 0
// CHECK-NEXT: [[CB_S_F3:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_S_F3_PTR]], align 4
// CHECK-NEXT: store <3 x float> [[CB_S_F3]], ptr [[S_F3_PTR]], align 4
// CHECK-NEXT: [[CB_S_A_PTR:%.*]] = getelementptr inbounds %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_S_PTR]], i32 0, i32 1
// CHECK-NEXT: [[S_A_PTR:%.*]] = getelementptr inbounds %struct.S, ptr %s, i32 0, i32 1
// CHECK-NEXT: [[CB_S_A:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_S_A_PTR]], align 4
// CHECK-NEXT: store i32 [[CB_S_A]], ptr [[S_A_PTR]], align 4
    S s = CB.s;

// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CB)
// CHECK-NEXT: [[CB_S_PTR:%.*]] = getelementptr inbounds nuw %MyConstants, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 11
// CHECK-NEXT: [[CB_S_F3_PTR:%.*]] = getelementptr inbounds nuw %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_S_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CB_S_F:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_S_F3_PTR]], align 4
// CHECK-NEXT: store <3 x float> [[CB_S_F]], ptr %f3, align 4
    float3 f3 = CB.s.f3;
}

// CHECK: define {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}})
// CHECK: [[HANDLE_PTR:%.*]] = getelementptr inbounds nuw %"class.hlsl::ConstantBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-DXIL: [[HANDLE:%.*]] = load target("dx.CBuffer", %MyConstants), ptr [[HANDLE_PTR]], align 4
// CHECK-DXIL: [[BASE_PTR:%.*]] = call ptr addrspace([[CONST_ADDR_SPACE]]) @llvm.dx.resource.getbasepointer.p2.tdx.CBuffer_s_MyConstantsst(target("dx.CBuffer", %MyConstants) [[HANDLE]])
// CHECK-SPV: [[HANDLE:%.*]] = load target("spirv.VulkanBuffer", %MyConstants, 2, 0), ptr [[HANDLE_PTR]], align 8
// CHECK-SPV: [[BASE_PTR:%.*]] = call ptr addrspace([[CONST_ADDR_SPACE]]) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_MyConstantss_2_0t(target("spirv.VulkanBuffer", %MyConstants, 2, 0) [[HANDLE]])
// CHECK: ret ptr addrspace([[CONST_ADDR_SPACE]]) [[BASE_PTR]]

// CHECK-LABEL: TestArrayAccess
void TestArrayAccess() {
// CHECK: [[TMP0:%.*]] = alloca %"class.hlsl::ConstantBuffer.0", align {{(4|8)}}
// CHECK: [[TMP1:%.*]] = alloca %"class.hlsl::ConstantBuffer.0", align {{(4|8)}}

// CHECK: call void @hlsl::ConstantBuffer<S>::__createFromImplicitBinding({{.*}})(ptr {{.*}} sret(%"class.hlsl::ConstantBuffer.0") align {{(4|8)}} [[TMP0]],
// CHECK-SAME: i32 noundef 1, i32 noundef 0, i32 noundef 2, i32 noundef 1, ptr noundef [[CBArrayStr]])
// CHECK-NEXT: [[CB_1_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<S>::operator S const AS[[CONST_ADDR_SPACE]]&() const(ptr{{.*}} [[TMP0]])
// CHECK-NEXT: [[CB_1_F3_PTR:%.*]] = getelementptr inbounds nuw %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_1_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CB_1_F3:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_1_F3_PTR]], align 4
// CHECK-NEXT: store <3 x float> [[CB_1_F3]], ptr %f3, align 4
    float3 f3 = CBArray[1].f3;

// CHECK: call void @hlsl::ConstantBuffer<S>::__createFromImplicitBinding({{.*}})(ptr {{.*}} sret(%"class.hlsl::ConstantBuffer.0") align {{(4|8)}} [[TMP1]],
// CHECK-SAME: i32 noundef 1, i32 noundef 0, i32 noundef 2, i32 noundef 0, ptr noundef [[CBArrayStr]])
// CHECK-NEXT: [[CB_0_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<S>::operator S const AS[[CONST_ADDR_SPACE]]&() const(ptr{{.*}} [[TMP1]])
// CHECK-NEXT: [[CB_0_A_PTR:%.*]] = getelementptr inbounds nuw %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_0_PTR]], i32 0, i32 1
// CHECK-NEXT: [[CB_0_A:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_0_A_PTR]], align 4
// CHECK-NEXT: store i32 [[CB_0_A]], ptr %a, align 4
    int a = CBArray[0].a;
}

// CHECK-DXIL: declare ptr addrspace([[CONST_ADDR_SPACE]]) @llvm.dx.resource.getbasepointer.p2.tdx.CBuffer_s_MyConstantsst(target("dx.CBuffer", %MyConstants))
// CHECK-SPV: declare ptr addrspace([[CONST_ADDR_SPACE]]) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_MyConstantss_2_0t(target("spirv.VulkanBuffer", %MyConstants, 2, 0))
