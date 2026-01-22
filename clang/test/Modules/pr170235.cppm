// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/lib.cppm -emit-module-interface -o %t/lib.pcm
// RUN: %clang_cc1 -std=c++20 %t/main.cpp -fmodule-file=lib=%t/lib.pcm -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/lib.cppm -emit-reduced-module-interface -o %t/lib.pcm
// RUN: %clang_cc1 -std=c++20 %t/main.cpp -fmodule-file=lib=%t/lib.pcm -fsyntax-only -verify

//--- lib.cppm
export module lib;
namespace lib {
    struct A;
    // Definition comes BEFORE the class declaration
    int foo(const A &, int) { return 42; }

    struct A {
        // Friend declaration inside the class
        friend int foo(const A &, int);
    };

    export A a{};
}
//--- main.cpp
// expected-no-diagnostics
import lib;
int main() {
    // Should be found via ADL since lib::a is of type lib::A
    auto res1 = foo(lib::a, 1); 
    return 0;
}
