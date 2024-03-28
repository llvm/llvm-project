// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// Make sure HLSLRootSignatureAttr is created with Var 'main.RS'

// CHECK: FunctionDecl 0x{{.*}} main 'void ()'
// CHECK-NEXT:   |-CompoundStmt
// CHECK-NEXT:   |-HLSLShaderAttr 0x{{.*}} Compute
// CHECK-NEXT:   |-HLSLRootSignatureAttr 0x{{.*}} "" Var 0x{{.*}} 'main.RS' 'main.RS'
// CHECK-NEXT:   `-HLSLNumThreadsAttr 0x{{.*}} 1 1 1

[shader("compute")]
[RootSignature("")]
[numthreads(1,1,1)]
void main() {}
