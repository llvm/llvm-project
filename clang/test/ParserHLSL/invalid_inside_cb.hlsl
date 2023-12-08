// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// template not allowed inside cbuffer.
cbuffer A {
    // expected-error@+2 {{invalid declaration inside cbuffer}}
    template<typename T>
    T foo(T t) { return t;}
}

cbuffer A {
    // expected-error@+2 {{invalid declaration inside cbuffer}}
    template<typename T>
    struct S { float s;};
}

// typealias not allowed inside cbuffer.
cbuffer A {
    // expected-error@+2 {{invalid declaration inside cbuffer}}
    // expected-warning@+1 {{alias declarations are a C++11 extension}}
    using F32 = float;
}
