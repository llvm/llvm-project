// Regression test for https://github.com/llvm/llvm-project/pull/237067:
// the parent map must contain the lambda body even when the closure type's
// call operator is a declaration without a body (merged from another module),
// otherwise this check hits llvm_unreachable.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang -std=c++20 -I%t --precompile %t/mod_a.cppm -o %t/mod_a.pcm
// RUN: %clang -std=c++20 -I%t --precompile %t/mod_b.cppm -o %t/mod_b.pcm
// RUN: clang-tidy -checks='-*,misc-static-initialization-cycle' %t/main.cppm -- \
// RUN:   -std=c++20 -x c++-module -fmodule-file=mod_a=%t/mod_a.pcm \
// RUN:   -fmodule-file=mod_b=%t/mod_b.pcm

//--- shared.h
#pragma once
struct SharedType {
  int a{};
};

template <class T>
inline constexpr unsigned long self = sizeof(T);

template <class T>
inline constexpr auto probe = [] { return self<T>; }();

//--- mod_a.cppm
module;
#include "shared.h"
export module mod_a;
export inline constexpr unsigned long kA = probe<SharedType>;

//--- mod_b.cppm
module;
#include "shared.h"
export module mod_b;
export inline constexpr unsigned long kB = probe<SharedType>;

//--- main.cppm
export module main;
import mod_a;
import mod_b;
