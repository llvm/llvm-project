// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang -std=c++20 %t/a.cppm --precompile -o %t/a.pcm
// RUN: %clang -std=c++20 %t/test.cc -fprebuilt-module-path=%t -fsyntax-only -Xclang -verify

//--- a.h
namespace ns {
namespace {
template <typename G> void func() {}
}
template <typename T = long> void a() { func<T>(); }
}

//--- a.cppm
module;
#include "a.h"
export module a;
export using ns::a;

//--- test.cc
import a;
auto m = (a(), 0);

// expected-no-diagnostics
