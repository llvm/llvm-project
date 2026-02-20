// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl202x -fnative-half-type -fnative-int16-type -verify -Wconversion %s

groupshared uint16_t SharedData;

void fn1(groupshared half Sh) {
// expected-note@-1{{candidate function not viable: no known conversion from 'groupshared uint16_t' (aka 'groupshared unsigned short') to 'groupshared half &' for 1st argument}}
  Sh = 5;
}

template<typename T>
void fnT(T A, T B) {
// expected-note@-1{{candidate function template not viable: no known conversion from 'groupshared uint16_t' (aka 'groupshared unsigned short') to 'groupshared half &' for 1st argument}}
  A = B;
}

void fn2() {
  fn1(SharedData);
  // expected-error@-1{{no matching function for call to 'fn1'}}
  // not sure why anyone would do thats but just making sure templates are sane
  fnT<groupshared half>(SharedData, SharedData);
  // expected-error@-1{{no matching function for call to 'fnT'}}
}
