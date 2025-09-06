// RUN: %clang_cc1 -fsyntax-only -verify -std=c++03 %s

struct S {
    S([[clang::lifetimebound]] int&) {}
};