// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -finclude-default-header -ast-dump -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-vertex -finclude-default-header -ast-dump -o - %s | FileCheck %s

// CHECK:      ParmVarDecl {{.*}} a 'float4':'vector<float, 4>'
// CHECK-NEXT: HLSLParsedSemanticAttr {{.*}} "ABC" 0
// CHECK-NEXT: HLSLAppliedSemanticAttr {{.*}} "ABC" 0

void main(float4 a : ABC) {
}
