// RUN: rm -rf %t
// RUN: split-file %s %t


// RUN: %clang_cc1 -std=c++20 -verify -emit-module-interface %t/B.cpp -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -verify -emit-module-interface %t/A.cpp -fmodule-file=A:B=%t/B.pcm -o %t/A.pcm

//--- A.cpp
// expected-no-diagnostics
export module A;
import :B;
export int x = dimensions + 1;

//--- B.cpp
// expected-no-diagnostics
export module A:B;
const int dimensions=3;
