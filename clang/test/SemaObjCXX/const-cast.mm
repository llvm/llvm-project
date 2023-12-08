// RUN: %clang_cc1 -fsyntax-only -verify %s

@class Foo;

void test() {
    const Foo *foo1 = 0;
    Foo *foo2 = foo1; // expected-error {{cannot initialize}}
}

void test1() {
    const Foo *foo1 = 0;
    Foo *foo2 = const_cast<Foo*>(foo1);
}
