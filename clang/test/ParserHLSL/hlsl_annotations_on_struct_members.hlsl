// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// TODO: update once we handle annotations on struct fields

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
