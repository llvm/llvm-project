// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/bar.cppm -emit-module-interface -o %t/bar.pcm
// RUN: %clang_cc1 -std=c++20 %t/foo.cc -fmodule-file=bar=%t/bar.pcm -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/bar.cppm -emit-module-interface \
// RUN:     -o %t/bar.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/foo.cc \
// RUN:     -fmodule-file=bar=%t/bar.pcm -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/bar.cppm -emit-reduced-module-interface -o %t/bar.pcm
// RUN: %clang_cc1 -std=c++20 %t/foo.cc -fmodule-file=bar=%t/bar.pcm -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/bar.cppm -emit-reduced-module-interface \
// RUN:     -o %t/bar.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/foo.cc \
// RUN:     -fmodule-file=bar=%t/bar.pcm -fsyntax-only -verify

//--- h.hpp
#pragma once

struct T {
    constexpr T(const char *) {}
};
template <char... c>
struct t {
    inline constexpr operator T() const { return {s}; }

private:
    inline static constexpr char s[]{c..., '\0'};
};

//--- bar.cppm
module;
#include "h.hpp"
export module bar;
export inline constexpr auto k = t<'k'>{};

//--- foo.cc
// expected-no-diagnostics
#include "h.hpp"
import bar;
void f() {
  T x = k;
}
