// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-reduced-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

//--- enum.h
enum E { Value };

//--- M.cppm
module;
#include "enum.h"
export module M;
auto e = Value;

//--- use.cpp
// expected-no-diagnostics
import M;
#include "enum.h"

auto e = Value;
