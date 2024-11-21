// RUN: %clang_cc1 -std=c++20 %s -verify -fsyntax-only

// expected-no-diagnostics
export module mod;
extern "C++" void func();
export extern "C++" {
    void func();
}
