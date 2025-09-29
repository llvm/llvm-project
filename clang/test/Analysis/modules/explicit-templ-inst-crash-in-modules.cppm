// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// DEFINE: %{common-flags}= -std=c++20 -I %t -fprebuilt-module-path=%t
//
// RUN: %clang_cc1 %{common-flags} %t/other.cppm -emit-module-interface -o %t/other.pcm
// RUN: %clang_analyze_cc1 -analyzer-checker=core %{common-flags} %t/entry.cppm -verify


//--- MyStruct.h
template <typename T> struct MyStruct {
  T data = 0;
};
template struct MyStruct<int>; // Explicit template instantiation.

//--- other.cppm
module;
#include "MyStruct.h"
export module other;
static void implicit_instantiate_MyStruct() {
  MyStruct<int> var;
  (void)var;
}

//--- entry.cppm
// expected-no-diagnostics
module;
#include "MyStruct.h"
module other;

void entry_point() {
  MyStruct<int> var; // no-crash
  (void)var;
}
