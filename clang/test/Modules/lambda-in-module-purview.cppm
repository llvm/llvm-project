// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/impl.cpp -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- a.cppm
export module a;
export auto cmp = [](auto l, auto r) {
  return l < r;
};

//--- use.cpp
// expected-no-diagnostics
import a;
auto x = cmp;

//--- impl.cpp
// expected-no-diagnostics
// The linkage of 'v' is queried to number the lambda before its type is
// deduced; the deduced type has internal linkage.
module a;
namespace { struct S {}; }
auto v = [] { return S{}; }();
