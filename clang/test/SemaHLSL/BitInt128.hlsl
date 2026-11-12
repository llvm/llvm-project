// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s -verify

// expected-error@+1 {{_BitInt is not supported on this target}}
_BitInt(128) i128;

// expected-error@+1 {{_BitInt is not supported on this target}}
unsigned _BitInt(128) u128;
