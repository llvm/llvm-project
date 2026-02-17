// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl202x -fnative-half-type -fnative-int16-type -verify -Wconversion %s

groupshared uint16_t SharedData;

void fn1(groupshared half Sh) {
// expected-note@-1{{candidate function not viable: cannot bind reference in generic address space to object in address space 'groupshared' in 1st argument}}
  Sh = 5;
}

void fn2() {
  fn1((half)SharedData);
  // expected-error@-1{{no matching function for call to 'fn1'}}
}
