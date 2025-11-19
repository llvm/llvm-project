// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/A.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.cpp -fmodule-file=A=%t/A.pcm -fsyntax-only -verify

//--- A.cppm
export module A;
export template <class T>
class C {};

export template <class T>
void foo() {
    C<T> value;
    (void) value;
}

//--- A.cpp
// expected-no-diagnostics
import A;
namespace {
class Local {};
}
void test() {
    foo<Local>();
}
