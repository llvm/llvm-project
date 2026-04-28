// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -fsyntax-only -verify %s

struct T { // expected-note 3 {{candidate constructor}}
    int a;
};

ConstantBuffer<T> c;

RWBuffer<int> b;

template<class Tm>
void foo(Tm t) {
    b[0] = t.a;
}

[numthreads(1,1,1)]
void main() {
    T t = c; // expected-error {{no viable constructor copying variable of type 'hlsl_constant T'}}
    foo(c);
}
