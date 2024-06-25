// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -fnative-half-type -verify %s

// expected-warning@+1{{cannot mix packoffset elements with nonpackoffset elements in a cbuffer}}
cbuffer Mix
{
    float4 M1 : packoffset(c0);
    float M2;
    float M3 : packoffset(c1.y);
}

// expected-warning@+1{{cannot mix packoffset elements with nonpackoffset elements in a cbuffer}}
cbuffer Mix2
{
    float4 M4;
    float M5 : packoffset(c1.y);
    float M6 ;
}

// expected-error@+1{{attribute 'packoffset' only applies to shader constant in a constant buffer}}
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

cbuffer Double {
    // expected-error@+1{{packoffset at 'y' not match alignment 64 required by 'double'}}
    double d : packoffset(c.y);
    // expected-error@+1{{packoffset cannot cross register boundary}}
	double2 d2 : packoffset(c.z);
    // expected-error@+1{{packoffset cannot cross register boundary}}
	double3 d3 : packoffset(c.z);
}

cbuffer ParsingFail {
// expected-error@+1{{expected identifier}}
float pf0 : packoffset();
// expected-error@+1{{expected identifier}}
float pf1 : packoffset((c0));
// expected-error@+1{{expected ')'}}
float pf2 : packoffset(c0, x);
// expected-error@+1{{invalid component 'X' used}}
float pf3 : packoffset(c.X);
// expected-error@+1{{expected '(' after ''}}
float pf4 : packoffset;
// expected-error@+1{{expected identifier}}
float pf5 : packoffset(;
// expected-error@+1{{expected '(' after '}}
float pf6 : packoffset);
// expected-error@+1{{expected '(' after '}}
float pf7 : packoffset c0.x;

// expected-error@+1{{invalid component 'xy' used}}
float pf8 : packoffset(c0.xy);
// expected-error@+1{{invalid component 'rg' used}}
float pf9 : packoffset(c0.rg);
// expected-error@+1{{invalid component 'yes' used}}
float pf10 : packoffset(c0.yes);
// expected-error@+1{{invalid component 'woo'}}
float pf11 : packoffset(c0.woo);
// expected-error@+1{{invalid component 'xr' used}}
float pf12 : packoffset(c0.xr);
}

struct ST2 {
  float a;
  float2 b;
};

cbuffer S {
  float S0 : packoffset(c0.y);
  ST2 S1[2] : packoffset(c1);
  // expected-error@+1{{packoffset overlap between 'S2', 'S1'}}
  half2 S2 : packoffset(c1.w);
  half2 S3 : packoffset(c2.w);
}

struct ST23 {
  float s0;
  ST2 s1;
};

cbuffer S2 {
  float S20 : packoffset(c0.y);
  ST2 S21 : packoffset(c1);
  half2 S22 : packoffset(c2.w);
  double S23[2] : packoffset(c3);
  // expected-error@+1{{packoffset overlap between 'S24', 'S23'}}
  float S24 : packoffset(c3.z);
  float S25 : packoffset(c4.z);
}
