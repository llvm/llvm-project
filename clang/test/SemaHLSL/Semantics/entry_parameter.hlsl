// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -hlsl-entry CSMain -x hlsl  -finclude-default-header  -ast-dump -o - %s | FileCheck %s

[numthreads(8,8,1)]
void CSMain(int GI : SV_GroupIndex, uint ID : SV_DispatchThreadID, uint GID : SV_GroupID, uint GThreadID : SV_GroupThreadID) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:6 CSMain 'void (int, uint, uint, uint)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:17 GI 'int'
// CHECK-NEXT: HLSLSV_GroupIndexAttr
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:42 ID 'uint'
// CHECK-NEXT: HLSLSV_DispatchThreadIDAttr
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:73 GID 'uint'
// CHECK-NEXT: HLSLSV_GroupIDAttr
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:96 GThreadID 'uint'
// CHECK-NEXT: HLSLSV_GroupThreadIDAttr
}
