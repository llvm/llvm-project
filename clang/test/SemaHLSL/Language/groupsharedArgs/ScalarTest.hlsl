// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl202x -fnative-half-type -fnative-int16-type -verify -Wconversion %s

groupshared uint16_t SharedData;

void fn1(groupshared half Sh) {
// expected-note@-1{{candidate function not viable: no known conversion from 'groupshared uint16_t' (aka 'groupshared unsigned short') to 'groupshared half &' for 1st argument}}
  Sh = 5;
}

void fn2() {
  fn1(SharedData);
  // expected-error@-1{{no matching function for call to 'fn1'}}
}
