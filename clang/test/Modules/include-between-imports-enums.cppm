// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-reduced-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

// Test that a textual #include sandwiched between two import declarations
// of modules that both include the same header in their GMFs does not lose
// enum declarations. See https://github.com/llvm/llvm-project/issues/188853

//--- enum.h
#ifndef ENUM_H
#define ENUM_H
namespace ns {
enum E { Value1, Value2, Value3 };
}
#endif

//--- A.cppm
module;
#include "enum.h"
export module A;
export auto a = ns::Value1;

//--- B.cppm
module;
#include "enum.h"
export module B;
export auto b = ns::Value2;

//--- use.cpp
// expected-no-diagnostics
import A;
#include "enum.h"
import B;

auto x = ns::Value3;

namespace ns {
auto y = Value1;
}
