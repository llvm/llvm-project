// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -finclude-default-header -std=hlsl202x -fnative-half-type -fnative-int16-type -verify -Wconversion %s

void fn1(groupshared half Sh) {
// expected-note@-1{{candidate function not viable: cannot bind reference in generic address space to object in address space 'groupshared' in 1st argument}}
  Sh = 5;
}

[numthreads(4, 1, 1)]
void main(uint3 TID : SV_GroupThreadID) {
  half tmp = 1.0;
  fn1(tmp);
  // expected-error@-1{{no matching function for call to 'fn1'}}
}
