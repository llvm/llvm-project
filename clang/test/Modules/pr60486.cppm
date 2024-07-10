// Address: https://github.com/llvm/llvm-project/issues/60486
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=a=%t/a.pcm %t/b.cppm -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=a=%t/a.pcm %t/b.cppm -fsyntax-only -verify

//--- foo.h
template<typename = void>
struct s {
};

template<typename>
concept c = requires { s{}; };

//--- a.cppm
module;
#include "foo.h"
export module a;

//--- b.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module b;
import a;
