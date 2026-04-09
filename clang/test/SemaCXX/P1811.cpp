// RUN: rm -rf %t
// RUN: split-file %s %t


// RUN: %clang_cc1 -std=c++20 -verify -emit-module-interface %t/mod.cpp -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -fmodule-file=M=%t/mod.pcm %t/main1.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -fmodule-file=M=%t/mod.pcm %t/main2.cpp

//--- mod.cpp
// expected-no-diagnostics
module;
#include "A.h"
export module M;
export A f() {return A{};}

//--- A.h
// expected-no-diagnostics
#ifndef X
#define X

struct A{};

#endif

//--- main1.cpp
// expected-no-diagnostics
#include "A.h"
import M;

extern "C++" int main() {
  A a;
}

//--- main2.cpp
// expected-no-diagnostics
import M;
#include "A.h"

extern "C++" int main() {
  A a;
}
