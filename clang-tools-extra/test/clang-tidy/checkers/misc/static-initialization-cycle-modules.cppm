// The check walks the body of a lambda that is part of a variable template
// specialization instantiated in a module. Building the parent map must not
// skip that body: the lambda body is reached through the call operator of the
// closure type, and when the closure type is merged with one from another
// module, the call operator the closure class refers to can be a declaration
// without a body, with the definition living on another declaration of the
// same entity.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang -std=c++20 -x c++-module -I%t -fmodule-output=%t/mod_a.pcm -c %t/mod_a.cppm -o %t/mod_a.o
// RUN: %clang -std=c++20 -x c++-module -I%t -fmodule-output=%t/mod_b.pcm -c %t/mod_b.cppm -o %t/mod_b.o
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
