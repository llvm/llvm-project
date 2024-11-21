// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header -verify -Wdouble-promotion -Wconversion %s

void OutVecFn(out float3) {}
void InOutVecFn(inout float3) {}

// Case 1: Calling out and inout parameters with types that cannot be
// back-converted. In HLSL 2021 and earlier this only occurs when passing scalar
// arguments to vector parameters because scalar->vector conversion is implicit,
// but vector->scalar is not.
void case1() {
  float f;
  int i;
  OutVecFn(f); // expected-error{{illegal scalar extension cast on argument f to out paramemter}}
  InOutVecFn(f); // expected-error{{illegal scalar extension cast on argument f to inout paramemter}}

  OutVecFn(i); // expected-error{{illegal scalar extension cast on argument i to out paramemter}}
  InOutVecFn(i); // expected-error{{illegal scalar extension cast on argument i to inout paramemter}}
}

// Case 2: Conversion warnings on argument initialization. Clang generates
// implicit conversion warnings only on the writeback conversion for `out`
// parameters since the parameter is not initialized from the argument. Clang
// generates implicit conversion warnings on both the parameter initialization
// and the writeback for `inout` parameters since the parameter is both copied
// in and out of the function.

void OutFloat(out float) {}
void InOutFloat(inout float) {}

void case2() {
  double f;
  OutFloat(f); // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  InOutFloat(f); // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}} expected-warning{{implicit conversion loses floating-point precision: 'double' to 'float'}}
}
