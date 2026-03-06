// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// This test verifies local arrays of resources in HLSL.

// CHECK: @_ZL1A = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @_ZL1B = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @_ZL1C = internal global %"class.hlsl::RWBuffer" poison, align 4

RWBuffer<float> A : register(u1);
RWBuffer<float> B : register(u2);
RWBuffer<float> C : register(u3);
RWStructuredBuffer<float> Out : register(u0);

// CHECK: define internal void @_Z4mainv()
// CHECK-NEXT: entry:
[numthreads(4,1,1)]
void main() {
// CHECK-NEXT:  %First = alloca [3 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT:  %Second = alloca [4 x %"class.hlsl::RWBuffer"], align 4
  RWBuffer<float> First[3] = { A, B, C };
  RWBuffer<float> Second[4];

// Verify initialization of First array from an initialization list
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %First, ptr {{.*}} @_ZL1A)
// CHECK-NEXT: %[[Ptr1:.*]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %First, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[Ptr1]], ptr {{.*}} @_ZL1B)
// CHECK-NEXT: %[[Ptr2:.*]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %First, i32 2
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1ERKS1_(ptr {{.*}} %[[Ptr2]], ptr {{.*}} @_ZL1C)

// Verify default initialization of Second array, which means there is a loop iterating
// over the array elements and calling the default constructor for each
// CHECK-NEXT: %[[ArrayBeginPtr:.*]] = getelementptr inbounds [4 x %"class.hlsl::RWBuffer"], ptr %Second, i32 0, i32 0
// CHECK-NEXT: %[[ArrayEndPtr:.*]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %[[ArrayBeginPtr]], i32 4
// CHECK-NEXT: br label %[[ArrayInitLoop:.*]]
// CHECK: [[ArrayInitLoop]]:
// CHECK-NEXT: %[[ArrayCurPtr:.*]] = phi ptr [ %[[ArrayBeginPtr]], %entry ], [ %[[ArrayNextPtr:.*]], %[[ArrayInitLoop]] ]
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1Ev(ptr {{.*}} %[[ArrayCurPtr]])
// CHECK-NEXT: %[[ArrayNextPtr]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %[[ArrayCurPtr]], i32 1
// CHECK-NEXT: %[[ArrayInitDone:.*]] = icmp eq ptr %[[ArrayNextPtr]], %[[ArrayEndPtr]]
// CHECK-NEXT: br i1 %[[ArrayInitDone]], label %[[AfterArrayInit:.*]], label %[[ArrayInitLoop]]
// CHECK: [[AfterArrayInit]]:

// Initialize First[2] with C
// CHECK: %[[Ptr3:.*]] = getelementptr inbounds [4 x %"class.hlsl::RWBuffer"], ptr %Second, i32 0, i32 2
// CHECK: call {{.*}} @_ZN4hlsl8RWBufferIfEaSERKS1_(ptr {{.*}} %[[Ptr3]], ptr {{.*}} @_ZL1C)
  Second[2] = C;

  // NOTE: _ZN4hlsl8RWBufferIfEixEj is the subscript operator for RWBuffer<float>

// get First[1][0] value
// CHECK: %[[First_1_Ptr:.*]] = getelementptr inbounds [3 x %"class.hlsl::RWBuffer"], ptr %First, i32 0, i32 1
// CHECK: %[[BufPtr1:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[First_1_Ptr]], i32 noundef 0)
// CHECK: %[[Value1:.*]] = load float, ptr %[[BufPtr1]], align 4

// get Second[2][0] value
// CHECK: %[[Second_2_Ptr:.*]] = getelementptr inbounds [4 x %"class.hlsl::RWBuffer"], ptr %Second, i32 0, i32 2
// CHECK: %[[BufPtr2:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[Second_2_Ptr]], i32 noundef 0)
// CHECK: %[[Value2:.*]] = load float, ptr %[[BufPtr2]], align 4

// add them
// CHECK: %{{.*}} = fadd {{.*}} float %[[Value1]], %[[Value2]]
  Out[0] = First[1][0] + Second[2][0];
}
