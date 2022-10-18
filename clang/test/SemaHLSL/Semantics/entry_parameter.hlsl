// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl  -finclude-default-header  -ast-dump -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-mesh -x hlsl -ast-dump  -finclude-default-header  -verify -o - %s

[numthreads(8,8,1)]
// expected-error@+2 {{attribute 'SV_GroupIndex' is unsupported in Mesh shaders, requires Compute}}
// expected-error@+1 {{attribute 'SV_DispatchThreadID' is unsupported in Mesh shaders, requires Compute}}
void CSMain(int GI : SV_GroupIndex, uint ID : SV_DispatchThreadID) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:6 CSMain 'void (int, uint)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:17 GI 'int'
// CHECK-NEXT: HLSLSV_GroupIndexAttr
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:42 ID 'uint'
// CHECK-NEXT: HLSLSV_DispatchThreadIDAttr
}
