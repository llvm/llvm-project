// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -finclude-default-header -o - %s -ast-dump | FileCheck %s

struct S {
  float4 f0 : SV_Position;
// CHECK: FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:10 f0 'float4':'vector<float, 4>'
// CHECK:   HLSLSV_PositionAttr 0x{{[0-9a-fA-F]+}} <col:15> <<<NULL>>> 0
  float4 f1 : SV_Position3;
// CHECK: FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:10 f1 'float4':'vector<float, 4>'
// CHECK:   HLSLSV_PositionAttr 0x{{[0-9a-fA-F]+}} <col:15> <<<NULL>>> 3 SemanticExplicitIndex
};

// FIXME(Keenuts): add mandatory output semantic once those are implemented.
float4 main(S s) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:8 main 'float4 (S)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:15 s 'S'

// CHECK: HLSLSV_PositionAttr 0x{{[0-9a-fA-F]+}} <{{.*}}> Field 0x{{[0-9a-fA-F]+}} 'f0' 'float4':'vector<float, 4>' 0
// CHECK: HLSLSV_PositionAttr 0x{{[0-9a-fA-F]+}} <{{.*}}> Field 0x{{[0-9a-fA-F]+}} 'f1' 'float4':'vector<float, 4>' 3 SemanticExplicitIndex
}
