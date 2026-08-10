// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=foo=%t/foo.pcm -verify -fsyntax-only

//--- a.hpp
#pragma once
#define A 43

//--- foo.cppm
module;
#include "a.hpp"
export module foo;
export constexpr auto B = A;

//--- use.cpp
// expected-no-diagnostics
import foo;
#include "a.hpp"

static_assert(A == 43);
static_assert(B == 43);

