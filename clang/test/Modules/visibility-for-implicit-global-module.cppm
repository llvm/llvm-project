// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.interface.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.impl.cc -fmodule-file=a:interface=%t/a.pcm \
// RUN:     -verify -fsyntax-only

//--- a.interface.cppm
export module a:interface;
extern "C++" constexpr int a = 43;

//--- a.impl.cc
// expected-no-diagnostics
module a:impl;
import :interface;
static_assert(a == 43);

