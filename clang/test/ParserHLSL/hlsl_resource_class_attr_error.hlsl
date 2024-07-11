// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -o - %s -verify

// expected-error@+1{{'resource_class' attribute takes one argument}}
struct [[hlsl::resource_class()]] Eg1 {
  int i;  
};

Eg1 e1;

// expected-error@+1{{invalid resource class 'gibberish' used; expected 'SRV', 'UAV', 'CBuffer', or 'Sampler'}}
struct [[hlsl::resource_class(gibberish)]] Eg2 {
  int i;  
};

Eg2 e2;

RWBuffer<int> In : register(u1);


[numthreads(1,1,1)]
void main() {
  In[0] = e1.i + e2.i;
}
