// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -finclude-default-header -o - %s | FileCheck %s

// CHECK: VarDecl {{.*}} t1 'hlsl::Texture2D<vector<float, 4>>':'hlsl::Texture2D<>'
Texture2D t1;

// CHECK: VarDecl {{.*}} t2 'Texture2D<float>':'hlsl::Texture2D<float>'
Texture2D<float> t2;

// CHECK: VarDecl {{.*}} t3 'Texture2D<float4>':'hlsl::Texture2D<>'
Texture2D<float4> t3;

void main() {
}
