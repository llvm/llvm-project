// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -finclude-default-header -fsyntax-only -verify %s

// expected-no-diagnostics

struct S {
    float a;
    
    float foo() {
        return a;
    };

    void bar() {
        a = 1.0;
    }
};

ConstantBuffer<S> CB;

[numthreads(4,1,1)]
void main() {
    // Calling non-const member function is allowed for parity with DXC.
    float tmp = CB.foo();

    // Even if it modifies the buffer, it's allowed in Sema (backended/validations will catch it).
    CB.bar();
}
