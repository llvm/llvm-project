// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s 

RWBuffer<float> Buffer1;
RWBuffer<vector<float, 4> > BufferArray[4];

[numthreads(1,1,1)]
void main() {
}

// CHECK: !hlsl.uavs = !{![[Single:[0-9]+]], ![[Array:[0-9]+]]}
// CHECK-DAG: ![[Single]] = !{ptr @"?Buffer1@@3V?$RWBuffer@M@hlsl@@A", !"RWBuffer<float>", i32 0}
// CHECK-DAG: ![[Array]] = !{ptr @"?BufferArray@@3PAV?$RWBuffer@T?$__vector@M$03@__clang@@@hlsl@@A", !"RWBuffer<vector<float, 4> >", i32 1}
