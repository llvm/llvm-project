// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// expected-error@+1{{unsupported resource register binding 't' on variable of type 'float'}}
float f1 : register(t9);

// expected-error@+1{{deprecated legacy bool constant register binding 'b' used. 'b' is only used for constant buffer resource binding}}
float f2 : register(b9);

// expected-error@+1{{deprecated legacy int constant register binding 'i' used}}
float f3 : register(i9);


cbuffer g_cbuffer1 {
// expected-error@+1{{register binding 'c' ignored inside cbuffer/tbuffer declarations; use pack_offset instead}}
    float f4 : register(c2);
};

tbuffer g_tbuffer1 {
// expected-error@+1{{register binding 'c' ignored inside cbuffer/tbuffer declarations; use pack_offset instead}}
    float f5 : register(c2);
};

cbuffer g_cbuffer2 {
// expected-error@+1{{register binding type 'b' not supported for variable of type 'float'}}
    float f6 : register(b2);
};

tbuffer g_tbuffer2 {
// expected-error@+1{{register binding type 'i' not supported for variable of type 'float'}}
    float f7 : register(i2);
};
