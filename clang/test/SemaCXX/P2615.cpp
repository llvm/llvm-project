// RUN: rm -rf %t
// RUN: split-file %s %t


// RUN: %clang_cc1 -std=c++20 -verify -fsyntax-only %t/A.cpp

//--- A.cpp
// expected-no-diagnostics
export module A;
export namespace N {int x = 42;}
export using namespace N;
