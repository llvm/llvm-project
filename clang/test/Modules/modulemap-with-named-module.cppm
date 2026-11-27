// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cc -fmodule-map-file=%t/a.cppm.modulemap -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- a.h
#pragma once
static_assert(false);

//--- a.cppm
export module a;
export struct A {};

//--- a.cc
// expected-no-diagnostics
#include "a.h"

A a;

//--- a.cppm.modulemap
module a {
    header "a.h"
}
