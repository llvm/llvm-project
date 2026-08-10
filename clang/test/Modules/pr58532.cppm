// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/interface.cppm -emit-module-interface \
// RUN:     -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/implementation.cpp -fmodule-file=m=%t/m.pcm \
// RUN:     -fsyntax-only -verify

// Test again with reduced BMI.
// RUN: %clang_cc1 -std=c++20 %t/interface.cppm -emit-reduced-module-interface \
// RUN:     -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/implementation.cpp -fmodule-file=m=%t/m.pcm \
// RUN:     -fsyntax-only -verify

//--- invisible.h
#pragma once // This breaks things.
const int kInvisibleSymbol = 0;
struct invisible_struct
{};
#define INVISIBLE_DEFINE

//--- visible.h
#include "invisible.h"
const int kSadlyUndeclaredSymbol = kInvisibleSymbol;
using unfortunately_still_invisible_struct = invisible_struct;
#ifndef INVISIBLE_DEFINE
#    error "Still not defined."
#endif

//--- interface.cppm
module;
#include "visible.h"
export module m;

//--- implementation.cpp
// expected-no-diagnostics
module;
#include "visible.h"
module m;
