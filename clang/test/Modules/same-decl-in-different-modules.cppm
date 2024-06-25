// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod1.cppm -emit-module-interface -o %t/mod1.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod2.cppm -emit-module-interface -o %t/mod2.pcm
// RUN: %clang_cc1 -std=c++20 %t/test.cc -fprebuilt-module-path=%t -fsyntax-only -verify

//--- mod1.cppm
export module mod1;
export int v;
export void func();
export class A {};
export template <class C>
struct S {};

//--- mod2.cppm
export module mod2;
export int v;
export void func();
export class A;
export template <class C>
struct S {};

//--- test.cc
import mod1;
import mod2;
void test() {
    int value = v;
    func();
    A a;
    S<int> s;
}

// expected-error@mod1.cppm:* {{declaration 'v' attached to named module 'mod1' can't be attached to other modules}}
// expected-note@mod2.cppm:* {{}}
// expected-error@mod1.cppm:* {{declaration 'func' attached to named module 'mod1' can't be attached to other modules}}
// expected-note@mod2.cppm:* {{}}
// expected-error@mod1.cppm:* {{declaration 'A' attached to named module 'mod1' can't be attached to other modules}}
// expected-note@mod2.cppm:* {{}}
// expected-error@mod1.cppm:* 1+{{declaration 'S' attached to named module 'mod1' can't be attached to other modules}}
// expected-note@mod2.cppm:* 1+{{}}
