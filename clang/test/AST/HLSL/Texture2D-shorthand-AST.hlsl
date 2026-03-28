// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -finclude-default-header -o - %s | FileCheck %s

// CHECK: VarDecl {{.*}} t1 'hlsl::Texture2D<vector<float, 4>>':'hlsl::Texture2D<>'
Texture2D t1;

// CHECK: VarDecl {{.*}} t1_explicit 'Texture2D<>':'hlsl::Texture2D<>'
Texture2D<> t1_explicit;

// CHECK: VarDecl {{.*}} t2 'Texture2D<float>':'hlsl::Texture2D<float>'
Texture2D<float> t2;

// CHECK: VarDecl {{.*}} t3 'Texture2D<float4>':'hlsl::Texture2D<>'
Texture2D<float4> t3;

// CHECK: TypedefDecl {{.*}} tex_alias 'hlsl::Texture2D<vector<float, 4>>':'hlsl::Texture2D<>'
typedef Texture2D tex_alias;

struct S {
  // CHECK: FieldDecl {{.*}} tex 'hlsl::Texture2D<vector<float, 4>>':'hlsl::Texture2D<>'
  Texture2D tex;
};

// CHECK: FunctionDecl {{.*}} foo 'hlsl::Texture2D<vector<float, 4>> (hlsl::Texture2D<vector<float, 4>>)'
// CHECK: ParmVarDecl {{.*}} p 'hlsl::Texture2D<vector<float, 4>>':'hlsl::Texture2D<>'
Texture2D foo(Texture2D p) {
  // CHECK: VarDecl {{.*}} local 'hlsl::Texture2D<vector<float, 4>>':'hlsl::Texture2D<>'
  Texture2D local;
  return local;
}

template<typename T>
void template_foo(T p) {
  // CHECK: VarDecl {{.*}} local 'hlsl::Texture2D<vector<float, 4>>':'hlsl::Texture2D<>'
  Texture2D local;
}

void main() {
  template_foo(1);
}
