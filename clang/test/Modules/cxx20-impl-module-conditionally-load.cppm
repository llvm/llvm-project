// From https://github.com/llvm/llvm-project/issues/57293
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cpp -fmodule-file=m=%t/m.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/m.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

//--- m.cppm
export module m;

//--- m.cpp
// expected-no-diagnostics
module m;
