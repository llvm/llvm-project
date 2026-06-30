// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -finclude-default-header -o - %s | FileCheck %s

// CHECK: VarDecl {{.*}} t1 'hlsl::Texture2DArray<vector<float, 4>>':'hlsl::Texture2DArray<>'
Texture2DArray t1;

// CHECK: VarDecl {{.*}} t1_explicit 'Texture2DArray<>':'hlsl::Texture2DArray<>'
Texture2DArray<> t1_explicit;

// CHECK: VarDecl {{.*}} t2 'Texture2DArray<float>':'hlsl::Texture2DArray<float>'
Texture2DArray<float> t2;

// CHECK: VarDecl {{.*}} t3 'Texture2DArray<float4>':'hlsl::Texture2DArray<>'
Texture2DArray<float4> t3;

// CHECK: TypedefDecl {{.*}} tex_alias 'hlsl::Texture2DArray<vector<float, 4>>':'hlsl::Texture2DArray<>'
typedef Texture2DArray tex_alias;

struct S {
  // CHECK: FieldDecl {{.*}} tex 'hlsl::Texture2DArray<vector<float, 4>>':'hlsl::Texture2DArray<>'
  Texture2DArray tex;
};

// CHECK: FunctionDecl {{.*}} foo 'hlsl::Texture2DArray<vector<float, 4>> (hlsl::Texture2DArray<vector<float, 4>>)'
// CHECK: ParmVarDecl {{.*}} p 'hlsl::Texture2DArray<vector<float, 4>>':'hlsl::Texture2DArray<>'
Texture2DArray foo(Texture2DArray p) {
  // CHECK: VarDecl {{.*}} local 'hlsl::Texture2DArray<vector<float, 4>>':'hlsl::Texture2DArray<>'
  Texture2DArray local;
  return local;
}

template<typename T>
void template_foo(T p) {
  // CHECK: VarDecl {{.*}} local 'hlsl::Texture2DArray<vector<float, 4>>':'hlsl::Texture2DArray<>'
  Texture2DArray local;
}

void main() {
  template_foo(1);
}
