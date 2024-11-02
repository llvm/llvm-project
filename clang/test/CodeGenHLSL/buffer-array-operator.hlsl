// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

const RWBuffer<float> In;
RWBuffer<float> Out;

void fn(int Idx) {
  Out[Idx] = In[Idx];
}

// This test is intended to verify reasonable code generation of the subscript
// operator. In this test case we should be generating both the const and
// non-const operators so we verify both cases.

// Non-const comes first.
// CHECK: ptr @"??A?$RWBuffer@M@hlsl@@QBAAAMI@Z"
// CHECK: %this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT: %h = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %this1, i32 0, i32 0
// CHECK-NEXT: %0 = load ptr, ptr %h, align 4
// CHECK-NEXT: %1 = load i32, ptr %Idx.addr, align 4
// CHECK-NEXT: %arrayidx = getelementptr inbounds float, ptr %0, i32 %1
// CHECK-NEXT: ret ptr %arrayidx

// Const comes next, and returns the pointer instead of the value.
// CHECK: ptr @"??A?$RWBuffer@M@hlsl@@QAAAAMI@Z"
// CHECK: %this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT: %h = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %this1, i32 0, i32 0
// CHECK-NEXT: %0 = load ptr, ptr %h, align 4
// CHECK-NEXT: %1 = load i32, ptr %Idx.addr, align 4
// CHECK-NEXT: %arrayidx = getelementptr inbounds float, ptr %0, i32 %1
// CHECK-NEXT: ret ptr %arrayidx
