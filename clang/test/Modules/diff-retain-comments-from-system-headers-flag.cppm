// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm -fretain-comments-from-system-headers
// RUN: %clang_cc1 -std=c++20 %t/b.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only

//--- a.cppm
export module a;

//--- b.cpp
// expected-no-diagnostics
import a;
