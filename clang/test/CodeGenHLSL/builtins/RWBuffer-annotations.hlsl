// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s 

RWBuffer<float> Buffer1;
RWBuffer<vector<float, 4> > BufferArray[4];

RWBuffer<float> Buffer2 : register(u3);
RWBuffer<vector<float, 4> > BufferArray2[4] : register(u4);

RWBuffer<float> Buffer3 : register(u3, space1);
RWBuffer<vector<float, 4> > BufferArray3[4] : register(u4, space1);



[numthreads(1,1,1)]
void main() {
}

// CHECK: !hlsl.uavs = !{![[Single:[0-9]+]], ![[Array:[0-9]+]], ![[SingleAllocated:[0-9]+]], ![[ArrayAllocated:[0-9]+]], ![[SingleSpace:[0-9]+]], ![[ArraySpace:[0-9]+]]}
// CHECK-DAG: ![[Single]] = !{ptr @"?Buffer1@@3V?$RWBuffer@M@hlsl@@A", !"RWBuffer<float>", i32 0, i32 10, i32 -1, i32 0}
// CHECK-DAG: ![[Array]] = !{ptr @"?BufferArray@@3PAV?$RWBuffer@T?$__vector@M$03@__clang@@@hlsl@@A", !"RWBuffer<vector<float, 4> >", i32 1, i32 10, i32 -1, i32 0}
// CHECK-DAG: ![[SingleAllocated]] = !{ptr @"?Buffer2@@3V?$RWBuffer@M@hlsl@@A", !"RWBuffer<float>", i32 2, i32 10, i32 3, i32 0}
// CHECK-DAG: ![[ArrayAllocated]] = !{ptr @"?BufferArray2@@3PAV?$RWBuffer@T?$__vector@M$03@__clang@@@hlsl@@A", !"RWBuffer<vector<float, 4> >", i32 3, i32 10, i32 4, i32 0}
// CHECK-DAG: ![[SingleSpace]] = !{ptr @"?Buffer3@@3V?$RWBuffer@M@hlsl@@A", !"RWBuffer<float>", i32 4, i32 10, i32 3, i32 1}
// CHECK-DAG: ![[ArraySpace]] = !{ptr @"?BufferArray3@@3PAV?$RWBuffer@T?$__vector@M$03@__clang@@@hlsl@@A", !"RWBuffer<vector<float, 4> >", i32 5, i32 10, i32 4, i32 1}
