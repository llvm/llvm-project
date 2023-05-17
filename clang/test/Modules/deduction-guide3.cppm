// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Templ.cppm -emit-module-interface -o %t/Templ.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only

//--- Templ.cppm
export module Templ;
template <typename T>
class Templ {
public:
    Templ(T a) {}
};

template<typename T>
Templ(T t) -> Templ<T>;

//--- Use.cpp
import Templ;
void func() {
    Templ t(5); // expected-error {{declaration of 'Templ' must be imported from module 'Templ' before it is required}}
                // expected-error@-1 {{unknown type name 'Templ'}}
                // expected-note@Templ.cppm:3 {{declaration here is not visible}}
}

