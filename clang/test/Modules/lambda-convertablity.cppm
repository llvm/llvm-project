// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -fprebuilt-module-path=%t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -fprebuilt-module-path=%t -fsyntax-only -verify

//--- A.cppm
export module a;

export auto f() {
    return [](){};
}

//--- B.cppm
// expected-no-diagnostics
export module b;
import a;

static_assert(__is_convertible_to(decltype(f()), decltype(f())));
