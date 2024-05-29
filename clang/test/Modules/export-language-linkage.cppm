// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -module-file-info %t/a.pcm | FileCheck %t/a.cppm
// RUN: %clang_cc1 -std=c++20 %t/b.cpp -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-module-interface -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 %t/d.cpp -fsyntax-only -verify -fmodule-file=c=%t/c.pcm

// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cpp -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -fsyntax-only -verify
// RUN: %clang_cc1 -module-file-info %t/a.pcm | FileCheck %t/a.cppm

//--- a.cppm
export module a;
export extern "C++" int foo() { return 43; }
export extern "C++" {
    int a();
    int b();
    int c();
}

export {
    extern "C++" void f1();
    extern "C++" void f2();
    extern "C++" void f3();
}

extern "C++" void unexported();

// CHECK: Sub Modules:
// CHECK-NEXT: Implicit Module Fragment '<implicit global>'

//--- b.cpp
import a;
int use() {
    a();
    b();
    c();
    f1();
    f2();
    f3();
    unexported(); // expected-error {{declaration of 'unexported' must be imported from module 'a' before it is required}}
                   // expected-note@a.cppm:15 {{declaration here is not visible}}
    return foo();
}

//--- c.cppm
// expected-no-diagnostics
export module c;
extern "C++" {
    export int f();
    int h();
}

extern "C++" export int g();

//--- d.cpp
import c;
int use() {
    return f() + g();
}

int use_of_nonexported() {
    return h(); // expected-error {{declaration of 'h' must be imported from module 'c' before it is required}}
                // expected-note@c.cppm:5 {{declaration here is not visible}}
}
