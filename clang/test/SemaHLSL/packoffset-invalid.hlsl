// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -verify %s

// expected-error@+1{{cannot mix packoffset elements with nonpackoffset elements in a cbuffer}}
cbuffer Mix
{
    float4 M1 : packoffset(c0);
    float M2;
    float M3 : packoffset(c1.y);
}

// expected-error@+1{{cannot mix packoffset elements with nonpackoffset elements in a cbuffer}}
cbuffer Mix2
{
    float4 M4;
    float M5 : packoffset(c1.y);
    float M6 ;
}

// expected-error@+1{{attribute 'packoffset' only applies to cbuffer constant}}
float4 g : packoffset(c0);

cbuffer IllegalOffset
{
    // expected-error@+1{{invalid resource class specifier 't2' for packoffset, expected 'c'}}
    float4 i1 : packoffset(t2);
    // expected-error@+1{{invalid component 'm' used; expected 'x', 'y', 'z', or 'w'}}
    float i2 : packoffset(c1.m);
}

cbuffer Overlap
{
    float4 o1 : packoffset(c0);
    // expected-error@+1{{packoffset overlap between 'o2', 'o1'}}
    float2 o2 : packoffset(c0.z);
}

cbuffer CrossReg
{
    // expected-error@+1{{packoffset cannot cross register boundary}}
    float4 c1 : packoffset(c0.y);
    // expected-error@+1{{packoffset cannot cross register boundary}}
    float2 c2 : packoffset(c1.w);
}

struct ST {
  float s;
};

cbuffer Aggregate
{
    // expected-error@+1{{packoffset cannot cross register boundary}}
    ST A1 : packoffset(c0.y);
    // expected-error@+1{{packoffset cannot cross register boundary}}
    float A2[2] : packoffset(c1.w);
}
