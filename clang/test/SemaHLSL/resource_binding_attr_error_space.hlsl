// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// valid
cbuffer cbuf {
    RWBuffer<int> r : register(u0, space0);
}

cbuffer cbuf2 {
    struct x {
        // expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
        RWBuffer<int> E : register(u2, space3);
    };
}

struct MyStruct {
    RWBuffer<int> E;
};

cbuffer cbuf3 {
  MyStruct E : register(u2, space3);
}

// expected-error@+1 {{invalid space specifier 's2' used; expected 'space' followed by an integer, like space1}}
cbuffer a : register(b0, s2) {

}

// expected-error@+1 {{invalid space specifier 'spaces' used; expected 'space' followed by an integer, like space1}}
cbuffer b : register(b2, spaces) {

}

// expected-error@+1 {{wrong argument format for hlsl attribute, use space3 instead}}
cbuffer c : register(b2, space 3) {}

// expected-error@+1 {{register space cannot be specified on global constants}}
int d : register(c2, space3);

// expected-error@+1 {{register space cannot be specified on global constants}}
int e : register(c2, space0);

// expected-error@+1 {{register space cannot be specified on global constants}}
int f : register(c2, space00);

// valid
RWBuffer<int> g : register(u2, space0);

// valid
RWBuffer<int> h : register(u2, space0);

struct S {
    RWBuffer<int> a;
};

// expected-error@+1 {{register space cannot be specified on global constants}}
S thing : register(u2, space2);
