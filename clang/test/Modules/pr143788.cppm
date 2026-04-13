// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/P.cppm -emit-module-interface -o %t/P.pcm
// RUN: %clang_cc1 -std=c++20 %t/I.cpp -fmodule-file=M:P=%t/P.pcm -fmodule-file=M=%t/M.pcm -fsyntax-only -verify

//--- H.hpp
struct S{};

//--- M.cppm
export module M;


//--- P.cppm
module;
#include "H.hpp"
module M:P;

using T = S;

//--- I.cpp
// expected-no-diagnostics
module M;
import :P;

T f() { return {}; }
