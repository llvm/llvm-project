// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

RasterizerOrderedBuffer<float> Buffer1;
RasterizerOrderedBuffer<vector<float, 4> > BufferArray[4];

RasterizerOrderedBuffer<float> Buffer2 : register(u3);
RasterizerOrderedBuffer<vector<float, 4> > BufferArray2[4] : register(u4);

RasterizerOrderedBuffer<float> Buffer3 : register(u3, space1);
RasterizerOrderedBuffer<vector<float, 4> > BufferArray3[4] : register(u4, space1);

void main() {}

// CHECK: !hlsl.uavs = !{![[Single:[0-9]+]], ![[Array:[0-9]+]], ![[SingleAllocated:[0-9]+]], ![[ArrayAllocated:[0-9]+]], ![[SingleSpace:[0-9]+]], ![[ArraySpace:[0-9]+]]}
// CHECK-DAG: ![[Single]] = !{ptr @Buffer1, i32 10, i32 9, i1 true, i32 -1, i32 0}
// CHECK-DAG: ![[Array]] = !{ptr @BufferArray, i32 10, i32 9, i1 true, i32 -1, i32 0}
// CHECK-DAG: ![[SingleAllocated]] = !{ptr @Buffer2, i32 10, i32 9, i1 true, i32 3, i32 0}
// CHECK-DAG: ![[ArrayAllocated]] = !{ptr @BufferArray2, i32 10, i32 9, i1 true, i32 4, i32 0}
// CHECK-DAG: ![[SingleSpace]] = !{ptr @Buffer3, i32 10, i32 9, i1 true, i32 3, i32 1}
// CHECK-DAG: ![[ArraySpace]] = !{ptr @BufferArray3, i32 10, i32 9, i1 true, i32 4, i32 1}
