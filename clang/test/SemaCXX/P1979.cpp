// RUN: rm -rf %t
// RUN: split-file %s %t


// RUN: %clang_cc1 -std=c++20 -verify -emit-module-interface %t/A.cpp -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -verify -emit-module-interface %t/myV.cpp -o %t/myV.pcm
// RUN: %clang_cc1 -std=c++20 -verify -emit-module-interface -fmodule-file=V=%t/myV.pcm %t/partition.cpp -o %t/partition.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -fmodule-file=V=%t/myV.pcm -fmodule-file=A=%t/A.pcm -fmodule-file=A:partition=%t/partition.pcm %t/interface.cpp

//--- A.cpp
// expected-no-diagnostics
export module A;

//--- myV.cpp
// expected-no-diagnostics
export module V;

export struct myV{};

//--- uses_vector.h
// expected-no-diagnostics
#ifndef x
#define x

import V;
#endif

//--- partition.cpp
// expected-no-diagnostics
module;
#include "uses_vector.h" // textually expands to import V;
module A:partition;

//--- interface.cpp
module A;
import :partition;
myV V; // expected-error {{declaration of 'myV' must be imported from module 'V' before it is required}}
       // expected-note@myV.cpp:4 {{declaration here is not visible}}
