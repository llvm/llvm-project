// RUN: %clang  --driver-mode=dxc -Zi -Fc - -T cs_6_6 -O0 %s | FileCheck %s

// CHECK: #dbg_declare(ptr [[ThisReg:%this\..*]], [[ThisMd:![0-9]+]],
// CHECK-DAG: [[ThisMd]] = !DILocalVariable(name: "this", arg: 1, scope: !{{[0-9]+}}, type: ![[type:[0-9]+]], flags: DIFlagArtificial | DIFlagObjectPointer)

RWBuffer<float4> Out : register(u7, space4);

[numthreads(8,1,1)]
void main(uint GI : SV_GroupIndex) {
  Out[GI] = 0;
}
