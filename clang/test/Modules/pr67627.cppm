// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -fmodule-file=A=%t/A.pcm -fsyntax-only -verify

//--- A.cppm
export module A;

//--- B.cppm
import A; // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}}
export module B; // expected-error {{module declaration must occur at the start of the translation unit}}
