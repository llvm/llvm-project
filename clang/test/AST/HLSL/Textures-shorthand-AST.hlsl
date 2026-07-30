// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -o - %s | FileCheck %s -DTEXTURE=Texture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -o - %s | FileCheck %s -DTEXTURE=Texture2DArray

// CHECK: VarDecl {{.*}} t1 'hlsl::[[TEXTURE]]<vector<float, 4>>':'hlsl::[[TEXTURE]]<>'
TEXTURE t1;

// CHECK: VarDecl {{.*}} t1_explicit '[[TEXTURE]]<>':'hlsl::[[TEXTURE]]<>'
TEXTURE<> t1_explicit;

// CHECK: VarDecl {{.*}} t2 '[[TEXTURE]]<float>':'hlsl::[[TEXTURE]]<float>'
TEXTURE<float> t2;

// CHECK: VarDecl {{.*}} t3 '[[TEXTURE]]<float4>':'hlsl::[[TEXTURE]]<>'
TEXTURE<float4> t3;

// CHECK: TypedefDecl {{.*}} tex_alias 'hlsl::[[TEXTURE]]<vector<float, 4>>':'hlsl::[[TEXTURE]]<>'
typedef TEXTURE tex_alias;

struct S {
  // CHECK: FieldDecl {{.*}} tex 'hlsl::[[TEXTURE]]<vector<float, 4>>':'hlsl::[[TEXTURE]]<>'
  TEXTURE tex;
};

// CHECK: FunctionDecl {{.*}} foo 'hlsl::[[TEXTURE]]<vector<float, 4>> (hlsl::[[TEXTURE]]<vector<float, 4>>)'
// CHECK: ParmVarDecl {{.*}} p 'hlsl::[[TEXTURE]]<vector<float, 4>>':'hlsl::[[TEXTURE]]<>'
TEXTURE foo(TEXTURE p) {
  // CHECK: VarDecl {{.*}} local 'hlsl::[[TEXTURE]]<vector<float, 4>>':'hlsl::[[TEXTURE]]<>'
  TEXTURE local;
  return local;
}

template<typename T>
void template_foo(T p) {
  // CHECK: VarDecl {{.*}} local 'hlsl::[[TEXTURE]]<vector<float, 4>>':'hlsl::[[TEXTURE]]<>'
  TEXTURE local;
}

void main() {
  template_foo(1);
}
