// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-mesh -x hlsl -ast-dump -verify -o - %s 

[numthreads(8,8, 1)]
// expected-error@+1 {{attribute 'SV_GroupIndex' is unsupported in Mesh shaders, requires Compute}}
void CSMain(int GI : SV_GroupIndex) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:6 CSMain 'void (int)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:17 GI 'int'
// CHECK-NEXT: HLSLSV_GroupIndexAttr
}
