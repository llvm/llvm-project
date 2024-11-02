// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
//--- templ.h
template <typename T, typename U = T>
class templ {};
template <typename T, typename U = void>
void templ_func() {}

//--- B.cppm
module;
#include "templ.h"
export module B;
export template <typename G>
templ<G> bar() {
  templ_func<G>();
  return {};
}

//--- Use.cpp
// expected-no-diagnostics
import B;
auto foo() {
  return bar<int>();
}
