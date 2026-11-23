// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/repro.cppm -fdeclspec -emit-module-interface -o %t/repro.pcm
// RUN: %clang_cc1 -std=c++20 %t/source.cpp -fdeclspec -fsyntax-only -verify -fprebuilt-module-path=%t

//--- repro_decl.hpp
#pragma once

extern "C"
{
    __declspec(selectany) int foo = 0;
}

//--- repro.cppm
module;
#include "repro_decl.hpp"

export module repro;

export inline int func()
{
    return foo;
}

//--- source.cpp
// expected-no-diagnostics
import repro;

#include "repro_decl.hpp"
