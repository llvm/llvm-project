// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

RWBuffer<float> Buffer;

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {}

// Make sure global variable for ctors/dtors removed.
// CHECK-NOT:@llvm.global_ctors
// CHECK-NOT:@llvm.global_dtors
//CHECK:      define void @main()
//CHECK-NEXT: entry:
//CHECK-NEXT:   call void @_GLOBAL__sub_I_GlobalConstructors.hlsl()
//CHECK-NEXT:   %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
//CHECK-NEXT:   call void @_Z4mainj(i32 %0)
//CHECK-NEXT:   ret void
