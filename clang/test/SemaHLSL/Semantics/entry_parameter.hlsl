// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -hlsl-entry CSMain -x hlsl  -finclude-default-header  -ast-dump -o - %s | FileCheck %s

[numthreads(8,8,1)]
void CSMain(int GI : SV_GroupIndex, uint ID : SV_DispatchThreadID, uint GID : SV_GroupID, uint GThreadID : SV_GroupThreadID) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:6 CSMain 'void (int, uint, uint, uint)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:17 GI 'int'
// CHECK-NEXT: HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:22> "SV_GroupIndex" 0
// CHECK-NEXT: HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:22> "SV_GroupIndex" 0
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:42 ID 'uint'
// CHECK-NEXT: HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:47> "SV_DispatchThreadID" 0
// CHECK-NEXT: HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:47> "SV_DispatchThreadID" 0
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:73 GID 'uint'
// CHECK-NEXT: HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:79> "SV_GroupID" 0
// CHECK-NEXT: HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:79> "SV_GroupID" 0
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:96 GThreadID 'uint'
// CHECK-NEXT: HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:108> "SV_GroupThreadID" 0
// CHECK-NEXT: HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:108> "SV_GroupThreadID" 0
}
