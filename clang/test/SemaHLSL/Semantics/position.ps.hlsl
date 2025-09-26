// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -finclude-default-header -o - %s -ast-dump | FileCheck %s

float4 main(float4 a : SV_Position) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:8 main 'float4 (float4)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:20 a 'float4':'vector<float, 4>'
// CHECK-NEXT: HLSLSV_PositionAttr 0x{{[0-9a-fA-F]+}} <{{.*}}>
}
