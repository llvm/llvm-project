// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -finclude-default-header -fsyntax-only -verify %s

struct S {
    float a;
    
    float foo() const {
        return a;
    };

    void bar() { // expected-note {{'bar' declared here}}
        a = 1.0;
    }
};

ConstantBuffer<S> CB;

[numthreads(4,1,1)]
void main() {
    // Bug: https://github.com/llvm/llvm-project/issues/153055
    // Calling non-const member function is allowed, but not  implemented yet.
    // We should remove the expected error when done.
    // expected-error@+1 {{cannot initialize object parameter of type 'const S' with an expression of type 'const hlsl_constant S'}}
    float tmp = CB.foo();

    // Calling non-const member function is not allowed.
    // expected-error@+1 {{'this' argument to member function 'bar' has type 'const hlsl_constant S', but function is not marked const}}
    CB.bar();
}
