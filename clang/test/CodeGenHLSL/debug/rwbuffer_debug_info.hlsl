// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s -debug-info-kind=standalone -dwarf-version=4  | FileCheck %s


// CHECK: [[DWTag:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "RWBuffer<float>", 
// CHECK: [[RWBuffer:![0-9]+]] = distinct !DISubprogram(name: "RWBuffer",
// CHECK-SAME: scope: [[DWTag]]
// CHECK: [[FirstThis:![0-9]+]] = !DILocalVariable(name: "this", arg: 1, scope: [[RWBuffer]], type: [[thisType:![0-9]+]]
// CHECK: [[thisType]] = !DIDerivedType(tag: DW_TAG_reference_type, baseType: [[DWTag]], size: 32)
RWBuffer<float> Out : register(u7, space4);

[numthreads(8,1,1)]
void main(int GI : SV_GroupIndex) {
  Out[GI] = 0;
}
