// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// tests that hlsl annotations are properly parsed when applied on field decls,
// and that the annotation gets properly placed on the AST.

struct Eg9{
  // CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} <col:1, col:8> col:8 implicit struct Eg9
  // CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:10:3, col:16> col:16 referenced a 'unsigned int'
  // CHECK: -HLSLSV_DispatchThreadIDAttr 0x{{[0-9a-f]+}} <col:20>
  unsigned int a : SV_DispatchThreadID;
};
Eg9 e9;


RWBuffer<int> In : register(u1);


[numthreads(1,1,1)]
void main() {
  In[0] = e9.a;
}
