// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header -verify %s

struct A {
  RWBuffer<float> Buf;
  RWBuffer<float> ManyBufs[5];
};

A array[10] : register(u10);

[numthreads(4,1,1)]
void main(uint GI : SV_GroupThreadID) {
  
  // expected-error@+1 {{index for struct array inside cbuffer that contains resources must be a constant integer expression}}
  array[GI].Buf[0] = 1.0f; 

  array[2].Buf[GI] = 2.0f; // ok

  // expected-error@+1 {{index for struct array inside cbuffer that contains resources must be a constant integer expression}}
  array[GI].ManyBufs[3][0] = 3.0f;

  array[1].ManyBufs[GI][0] = 4.0f; // ok

  array[1+1].ManyBufs[GI][0] = 4.0f; // ok

  int x = 3;
  array[1].ManyBufs[x][0] = 4.0f; // ok
}
