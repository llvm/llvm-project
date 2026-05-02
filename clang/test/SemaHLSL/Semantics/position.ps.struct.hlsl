// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -finclude-default-header -o - %s -ast-dump | FileCheck %s

struct S {
  float4 f0 : SV_Position;
// CHECK: FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:10 f0 'float4':'vector<float, 4>'
// CHECK-NEXT:  HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:15> "SV_Position" 0
  float4 f1 : SV_Position3;
// CHECK: FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:10 referenced f1 'float4':'vector<float, 4>'
// CHECK-NEXT:  HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:15> "SV_Position" 3
};

// FIXME(Keenuts): add mandatory output semantic once those are implemented.
float4 main(S s) : B {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:8 main 'float4 (S)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:15 used s 'S'
// CHECK-NEXT:  HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <line:4:15> "SV_Position" 0
// CHECK-NEXT:  HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <line:7:15> "SV_Position" 3

// CHECK:       HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:20> "B" 0
  return s.f1;
}
