// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mymod.cppm -emit-module-interface -o %t/mymod.pcm
// RUN: %clang_cc1 -std=c++20 %t/consumer.cpp -fprebuilt-module-path=%t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/mymod.cppm -emit-reduced-module-interface -o %t/mymod.pcm
// RUN: %clang_cc1 -std=c++20 %t/consumer.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/mymod.cppm -emit-module-interface -o %t/mymod.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/consumer.cpp -fprebuilt-module-path=%t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/mymod.cppm -emit-reduced-module-interface -o %t/mymod.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/consumer.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- r.h
struct F {
  template <typename... T> requires ((sizeof(T) > 0) && ...)
  void operator()(T...) {}
} f;

//--- mymod.cppm
module;
#include "r.h"
export module mymod;
export using ::f;

//--- consumer.cpp
// expected-no-diagnostics
void operator&&(struct X, struct X);
#include "r.h"
import mymod;

void g() { f(); }
