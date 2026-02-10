// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// This test verifies handling of multi-dimensional local arrays of resources
// when used as a function argument and local variable.

// CHECK: @_ZL1A = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @_ZL1B = internal global %"class.hlsl::RWBuffer" poison, align 4

RWBuffer<float> A : register(u10);
RWBuffer<float> B : register(u20);
RWStructuredBuffer<float> Out;

// NOTE: _ZN4hlsl8RWBufferIfEixEj is the subscript operator for RWBuffer<float> and
//       _ZN4hlsl18RWStructuredBufferIfEixEj is the subscript operator for RWStructuredBuffer<float>

// CHECK: define {{.*}} float @_Z3fooA2_A2_N4hlsl8RWBufferIfEE(ptr noundef byval([2 x [2 x %"class.hlsl::RWBuffer"]]) align 4 %Arr)
// CHECK-NEXT: entry:
float foo(RWBuffer<float> Arr[2][2]) {
// CHECK-NEXT: %[[Arr_1_Ptr:.*]] = getelementptr inbounds [2 x [2 x %"class.hlsl::RWBuffer"]], ptr %Arr, i32 0, i32 1
// CHECK-NEXT: %[[Arr_1_1_Ptr:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %[[Arr_1_Ptr]], i32 0, i32 1
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[Arr_1_1_Ptr]], i32 noundef 0)
// CHECK-NEXT: %[[Value:.*]] = load float, ptr %[[BufPtr]], align 4
// CHECK-NEXT: ret float %[[Value]]
  return Arr[1][1][0];
}

// CHECK: define internal void @_Z4mainv()
// CHECK-NEXT: entry:
[numthreads(4,1,1)]
void main() {
// CHECK: %L = alloca [2 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK: %[[ref_tmp:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[ref_tmp1:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[ref_tmp2:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[ref_tmp3:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[ref_tmp4:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[ref_tmp5:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[ref_tmp6:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[ref_tmp7:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[agg_tmp:.*]] = alloca [2 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[ref_tmp]], ptr {{.*}} @_ZL1A)
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[ref_tmp1]], ptr {{.*}} %[[ref_tmp]])
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[ref_tmp2]], ptr {{.*}} @_ZL1B)
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[ref_tmp3]], ptr {{.*}} %[[ref_tmp2]])
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[ref_tmp4]], ptr {{.*}} @_ZL1A)
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[ref_tmp5]], ptr {{.*}} %[[ref_tmp4]])
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[ref_tmp6]], ptr {{.*}} @_ZL1B)
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[ref_tmp7]], ptr {{.*}} %[[ref_tmp6]])
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %L, ptr {{.*}} %[[ref_tmp1]])
// CHECK-NEXT: %[[arrayinit_element:.*]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %L, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[arrayinit_element]], ptr {{.*}} %[[ref_tmp3]])
// CHECK-NEXT: %[[arrayinit_element8:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %L, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[arrayinit_element8]], ptr {{.*}} %[[ref_tmp5]])
// CHECK-NEXT: %[[arrayinit_element9:.*]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %[[arrayinit_element8]], i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[arrayinit_element9]], ptr {{.*}} %[[ref_tmp7]])
  RWBuffer<float> L[2][2] = { { A, B }, { A, B } };

// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[agg_tmp]], ptr align 4 %L, i32 16, i1 false)
// CHECK-NEXT: %[[ReturnedValue:.*]] = call {{.*}}float @_Z3fooA2_A2_N4hlsl8RWBufferIfEE(ptr noundef byval([2 x [2 x %"class.hlsl::RWBuffer"]]) align 4 %[[agg_tmp]])
// CHECK-NEXT: %[[OutBufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl18RWStructuredBufferIfEixEj(ptr {{.*}} @_ZL3Out, i32 noundef 0)
// CHECK-NEXT: store float %[[ReturnedValue]], ptr %[[OutBufPtr]], align 4
// CHECK-NEXT: ret void
  Out[0] = foo(L);
}
