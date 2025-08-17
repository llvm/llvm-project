// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -o - -fsyntax-only %s -verify

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
    // expected-error@+1 {{invalid declaration inside cbuffer}}
    using F32 = float;
}
