// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// expected-error@+1{{binding type 't' only applies to SRV resources}}
float f1 : register(t0);

// expected-error@+1 {{binding type 'u' only applies to UAV resources}}
float f2 : register(u0);

// expected-error@+1{{binding type 'b' only applies to constant buffers. The 'bool constant' binding type is no longer supported}}
float f3 : register(b9);

// expected-error@+1 {{binding type 's' only applies to sampler state}}
float f4 : register(s0);

// expected-error@+1{{binding type 'i' ignored. The 'integer constant' binding type is no longer supported}}
float f5 : register(i9);

// expected-error@+1{{binding type 'x' is invalid}}
float f6 : register(x9);

cbuffer g_cbuffer1 {
// expected-error@+1{{binding type 'c' ignored in buffer declaration. Did you mean 'packoffset'?}}
    float f7 : register(c2);
};

tbuffer g_tbuffer1 {
// expected-error@+1{{binding type 'c' ignored in buffer declaration. Did you mean 'packoffset'?}}
    float f8 : register(c2);
};

cbuffer g_cbuffer2 {
// expected-error@+1{{binding type 'b' only applies to constant buffer resources}}
    float f9 : register(b2);
};

tbuffer g_tbuffer2 {
// expected-error@+1{{binding type 'i' ignored. The 'integer constant' binding type is no longer supported}}
    float f10 : register(i2);
};

// expected-error@+1{{binding type 'c' only applies to numeric variables in the global scope}}
RWBuffer<float> f11 : register(c3);
