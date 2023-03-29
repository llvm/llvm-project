// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/M-A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-module-interface -o %t/M-B.pcm
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -o %t/M.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/Use.cpp -fmodule-file=M=%t/M.pcm -fsyntax-only \
// RUN:     -verify

//--- foo.h
template <typename T>
class Templ {
public:
    Templ(T a) {}
};

//--- A.cppm
module;
#include "foo.h"
export module M:A;
export using ::Templ;

//--- B.cppm
module;
#include "foo.h"
export module M:B;

//--- M.cppm
export module M;
export import :A;
export import :B;

//--- Use.cpp
// expected-no-diagnostics
import M;

void func() {
    Templ t(5);
}
