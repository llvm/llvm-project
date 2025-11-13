// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -finclude-default-header -o - %s -ast-dump | FileCheck %s

struct A {
  float4 x : A;
// CHECK: FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:10 x 'float4':'vector<float, 4>'
// CHECK-NEXT:  HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:14> "A" 0
};

struct Top {
  A f0 : B;
// CHECK: FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:5 f0 'A'
// CHECK-NEXT:  HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:10> "B" 0
  A f1 : C;
// CHECK: FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:5 f1 'A'
// CHECK-NEXT:  HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:10> "C" 0
};


// FIXME(Keenuts): add mandatory output semantic once those are implemented.
float4 main(Top s : D) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:8 main 'float4 (Top)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:17 s 'Top'
// CHECK-NEXT:  HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:21> "D" 0
// CHECK-NEXT:  HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:21> "D" 0
// CHECK-NEXT:  HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:21> "D" 1
}
