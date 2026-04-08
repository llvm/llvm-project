// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -fsyntax-only -verify %s

// expected-no-diagnostics

struct T {
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
    T t = c;
    foo(c);
}
