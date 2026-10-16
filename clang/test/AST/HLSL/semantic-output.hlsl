// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -finclude-default-header -ast-dump -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-vertex -finclude-default-header -ast-dump -o - %s | FileCheck %s

// CHECK: FunctionDecl {{.*}} main 'uint ()'
// CHECK:  HLSLParsedSemanticAttr {{.*}} "ABC" 0
// CHECK:  HLSLAppliedSemanticAttr {{.*}} "ABC" 0
uint main() : ABC {
  return 0;
}
