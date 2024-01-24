// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/test.cpp -fsyntax-only -verify

//--- i.h
#ifndef I_H
#pragma once
struct S{};
#endif

//--- test.cpp
// expected-no-diagnostics
#include "i.h"

int foo() {
    return sizeof(S);
}
