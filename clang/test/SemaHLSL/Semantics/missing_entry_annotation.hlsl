// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -hlsl-entry main -verify %s

[numthreads(1,1, 1)]
void main(int GI) { } // expected-error{{semantic annotations must be present for all parameters of an entry function or patch constant function}} expected-note{{'GI' declared here}}
