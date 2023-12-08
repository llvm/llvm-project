// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// expected-note@+1 {{declared here}}
cbuffer a {
  int x;
};

int foo() {
  // expected-error@+1 {{'a' does not refer to a value}}
  return sizeof(a);
}

// expected-error@+1 {{expected unqualified-id}}
template <typename Ty> cbuffer a { Ty f; };

// For back-compat reason, it is OK for multiple cbuffer/tbuffer use same name in hlsl.
// And these cbuffer name only used for reflection, cannot be removed.
cbuffer A {
  float A;
}

cbuffer A {
  float b;
}

tbuffer A {
  float a;
}

float bar() {
  // cbuffer/tbuffer name will not conflict with other variables.
  return A;
}

cbuffer a {
  // expected-error@+2 {{unknown type name 'oh'}}
  // expected-error@+1 {{expected ';' after top level declarator}}
  oh no!
  // expected-warning@+1 {{missing terminating ' character}}
  this isn't even valid HLSL code
  despite seeming totally reasonable
  once you understand that HLSL
  is so flaming weird.
}

tbuffer B {
  // expected-error@+1 {{unknown type name 'flaot'}}
  flaot f;
}
