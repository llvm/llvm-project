// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Templ.cppm -emit-module-interface -o %t/Templ.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only

// RUN: %clang_cc1 -std=c++20 %t/Templ.cppm -emit-reduced-module-interface -o %t/Templ.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only

//--- Templ.cppm
export module Templ;
export template <typename T>
class Templ {
public:
    Templ(T a) {}
};

template<typename T>
Templ(T t) -> Templ<T>;

//--- Use.cpp
// expected-no-diagnostics
import Templ;
void func() {
    Templ t(5);
}

