// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
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
// CHECK-NEXT: Implicit Module Fragment '<exported implicit global>'
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
    unexported(); // expected-error {{missing '#include'; 'unexported' must be declared before it is used}}
                   // expected-note@a.cppm:15 {{declaration here is not visible}}
    return foo();
}

//--- c.cppm
export module c;
extern "C++" {
    // We can't use `export` in an unnamed module.
    export int f(); // expected-error {{export declaration can only be used within a module purview}}
}

extern "C++" export int g(); // expected-error {{export declaration can only be used within a module purview}}
