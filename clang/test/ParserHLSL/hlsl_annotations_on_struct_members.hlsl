// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// previously, this test would result in an error shown below on the line that 
// declares variable a in struct Eg9:
// error: use of undeclared identifier
//     'SV_DispatchThreadID'
// This is because the annotation is parsed as if it was a c++ bit field, and an identifier
// that represents an integer is expected, but not found.

// This test ensures that hlsl annotations are attempted to be parsed when parsing struct decls.
// Ideally, we'd validate this behavior by ensuring the annotation is parsed and properly
// attached as an attribute to the member in the struct in the AST. However, in this case
// this can't happen presently because there are other issues with annotations on field decls.
// This test just ensures we make progress by moving the validation error from the realm of
// C++ and expecting bitfields, to HLSL and a specialized error for the recognized annotation.

struct Eg9{
// expected-error@+1{{attribute 'SV_DispatchThreadID' only applies to parameter}}
  int a : SV_DispatchThreadID;
};
Eg9 e9;


RWBuffer<int> In : register(u1);


[numthreads(1,1,1)]
void main() {
  In[0] = e9.a;
}
