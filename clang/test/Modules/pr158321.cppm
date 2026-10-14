// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/consumer.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- repro_header.h
namespace n
{
}

//--- m.cppm
module;
#include "repro_header.h"
export module m;
namespace n
{
    int x;
}

//--- consumer.cpp
// expected-no-diagnostics
import m;
namespace n
{
}