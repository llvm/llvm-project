// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic

void voidfn();

class Foo{};

int fun() {
    int x = 0;
    __builtin_static_analysis_assume(true);
    __builtin_static_analysis_assume(x <= 0);
    __builtin_static_analysis_assume(voidfn());  // expected-error{{cannot initialize a parameter of type 'bool' with an rvalue of type 'void}}
    __builtin_static_analysis_assume(Foo());  // expected-error{{no viable conversion from 'Foo' to 'bool'}}
    return x;
}
