// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -finclude-default-header -o - %s -ast-dump | FileCheck %s

// FIXME(Keenuts): change output semantic to something valid for pixels shaders
float4 main(float4 a : SV_Position2) : A {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:8 main 'float4 (float4)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:20 used a 'float4':'vector<float, 4>'
// CHECK-NEXT:  HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <col:24> "SV_Position" 2
// CHECK-NEXT:  HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:24> "SV_Position" 2

// CHECK:       HLSLParsedSemanticAttr 0x{{[0-9a-f]+}} <line:4:40> "A" 0
// CHECK:       HLSLAppliedSemanticAttr 0x{{[0-9a-f]+}} <col:40> "A" 0
  return a;
}
