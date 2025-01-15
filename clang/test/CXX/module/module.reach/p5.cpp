// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/B.cppm -fsyntax-only -verify
//
//--- A.cppm
export module A;
struct X {};
export using Y = X;

//--- B.cppm
export module B;
import A;
Y y; // OK, definition of X is reachable
X x; // expected-error {{unknown type name 'X'}}
