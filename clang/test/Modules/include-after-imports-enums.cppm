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
#pragma once

enum E { Value };

typedef enum { TypedefValue } TypedefEnum;
void useTypedefEnum(TypedefEnum);

//--- M.cppm
module;
#include "enum.h"
export module M;
auto e = Value;
export TypedefEnum typedefEnum;

//--- use.cpp
// expected-no-diagnostics
import M;
#include "enum.h"

auto e = Value;
static_assert(__is_same(decltype(TypedefValue), TypedefEnum));
inline void testTypedefEnum() { useTypedefEnum(TypedefValue); }
