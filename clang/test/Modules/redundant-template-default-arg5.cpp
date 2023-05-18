// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodule-name=mod -xc++ -emit-module %t/mod.cppmap -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodule-file=%t/mod.pcm -fsyntax-only %t/use.cc -verify

//--- mod.cppmap
module "mod" {
  export *
  header "mod.h"
}

//--- mod.h
#ifndef MOD
#define MOD
#include "templ.h"
#endif

//--- templ.h
#ifndef TEMPL
#define TEMPL
template <typename t1 = void>
inline constexpr bool inl = false;
#endif

//--- use.cc
// expected-no-diagnostics
#include "templ.h"
#include "mod.h"
