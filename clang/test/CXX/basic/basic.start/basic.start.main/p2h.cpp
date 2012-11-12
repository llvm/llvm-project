// RUN: %clang_cc1 -fsyntax-only -verify %s 

template<typename T>
int main() { } // expected-error{{'main' cannot be a template}}

