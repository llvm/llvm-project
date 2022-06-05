// This tests that we could use C++20 modules standalone.
// RUN: %clang -std=c++03 -fcxx-modules -fsyntax-only -Xclang -verify %s
// RUN: %clang -std=c++11 -fcxx-modules -fsyntax-only -Xclang -verify %s
// RUN: %clang -std=c++14 -fcxx-modules -fsyntax-only -Xclang -verify %s
// RUN: %clang -std=c++17 -fcxx-modules -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
export module M;
