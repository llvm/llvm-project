// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use.cppm -verify -fsyntax-only

// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use.cppm -verify -fsyntax-only

//--- foo.h
template <typename T, typename U = int>
class Templ;

template <typename T, typename U>
class Templ {
public:
    Templ(T t) {}
};

template <typename T>
Templ(T t) -> Templ<T, int>;

//--- A.cppm
module;
#include "foo.h"
export module A;

//--- Use.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module X;
import A;
void foo() {
    Templ t(0);
}
