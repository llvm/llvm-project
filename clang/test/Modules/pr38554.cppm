// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/c.cppm

//--- a.hpp
#pragma once
using a = int;

//--- b.hpp
#pragma once
#include "a.hpp"
a b;

//--- c.cppm
// expected-no-diagnostics
module;
#include "b.hpp"
export module c;
export using ::a;
