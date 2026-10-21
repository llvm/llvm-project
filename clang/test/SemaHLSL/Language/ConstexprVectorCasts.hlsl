// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -finclude-default-header -fnative-int16-type -fnative-half-type -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -finclude-default-header -fnative-int16-type -fnative-half-type                                         -verify=ref,both %s

const float3 F;

struct S {
  float V;
};

[numthreads(1,1,1)]
void main () {
  S Vals[] = {1,2,3}; // expected-note {{declared here}}
  constexpr int3 I = false ? (float3)Vals : Vals[0].V.xxx; // #conversions
  // both-warning@#conversions {{implicit conversion turns floating-point number into integer: 'float3' (aka 'vector<float, 3>') to 'vector<int, 3>' (vector of 3 'int' values)}}
  // both-warning@#conversions {{implicit conversion turns floating-point number into integer: 'vector<float, 3>' (vector of 3 'float' values) to 'vector<int, 3>' (vector of 3 'int' values)}}
  // both-error@#conversions {{constexpr variable 'I' must be initialized by a constant expression}}
  // expected-note@#conversions {{read of non-constexpr variable 'Vals' is not allowed in a constant expression}}

  constexpr float2 F2 = {4800000, -4800000};
  constexpr vector<int16_t,2> I16 = F2; // #range
  // both-warning@#range {{implicit conversion turns floating-point number into integer: 'const float2' (aka 'const vector<float, 2>') to 'vector<int16_t, 2>' (vector of 2 'int16_t' values)}}
  // both-error@#range {{constexpr variable 'I16' must be initialized by a constant expression}}
  // ref-note@#range {{value 4.8E+6 is outside the range of representable values of type 'int16_t' (aka 'short')}}
  // expected-note@#range {{value 4.8E+6 is outside the range of representable values of type 'vector<int16_t, 2>' (vector of 2 'int16_t' values)}}
}
