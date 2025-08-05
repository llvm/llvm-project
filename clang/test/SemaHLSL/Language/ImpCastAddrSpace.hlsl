// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -Wconversion -fnative-half-type %s -verify

static double D = 2.0;
static int I = D; // expected-warning{{implicit conversion turns floating-point number into integer: 'double' to 'int'}}
groupshared float F = I; // expected-warning{{implicit conversion from 'int' to 'float' may lose precision}}

export void fn() {
  half d = I; // expected-warning{{implicit conversion from 'int' to 'half' may lose precision}}
  int i = D; // expected-warning{{implicit conversion turns floating-point number into integer: 'double' to 'int'}}
  int j = F; // expected-warning{{implicit conversion turns floating-point number into integer: 'float' to 'int'}}
  int k = d; // expected-warning{{implicit conversion turns floating-point number into integer: 'half' to 'int'}}
}
