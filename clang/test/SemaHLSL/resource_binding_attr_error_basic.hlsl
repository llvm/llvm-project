// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// expected-error@+1{{unsupported resource register binding 't' on variable of type 'float'}}
float f1 : register(t0);


float f2 : register(c0);

// expected-error@+1{{deprecated legacy bool constant register binding 'b' used. 'b' is only used for constant buffer resource binding}}
float f3 : register(b9);

// expected-error@+1{{deprecated legacy int constant register binding 'i' used}}
float f4 : register(i9);

// expected-error@+1{{invalid register type 'x' used; expected 't', 'u', 'b', or 's'}}
float f5 : register(x9);

cbuffer g_cbuffer1 {
// expected-error@+1{{register binding 'c' ignored inside cbuffer/tbuffer declarations; use pack_offset instead}}
    float f6 : register(c2);
};

tbuffer g_tbuffer1 {
// expected-error@+1{{register binding 'c' ignored inside cbuffer/tbuffer declarations; use pack_offset instead}}
    float f7 : register(c2);
};

cbuffer g_cbuffer2 {
// expected-error@+1{{register binding type 'b' not supported for variable of type 'float'}}
    float f8 : register(b2);
};

tbuffer g_tbuffer2 {
// expected-error@+1{{register binding type 'i' not supported for variable of type 'float'}}
    float f9 : register(i2);
};

// expected-error@+1{{uav type 'RWBuffer<float>' requires register type 'u', but register type 'c' was used}}
RWBuffer<float> f10 : register(c3);