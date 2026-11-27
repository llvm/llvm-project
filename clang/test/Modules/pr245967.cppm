// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/err.cppm -I%t -emit-module-interface -o %t/err.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -I%t -fmodule-file=err=%t/err.pcm \
// RUN:   -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/use_extra.cpp -I%t -fmodule-file=err=%t/err.pcm \
// RUN:   -fsyntax-only -verify
//
// Test again with reduced BMI.
// RUN: %clang_cc1 -std=c++20 %t/err.cppm -I%t -emit-reduced-module-interface \
// RUN:   -o %t/err.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -I%t -fmodule-file=err=%t/err.pcm \
// RUN:   -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/use_extra.cpp -I%t -fmodule-file=err=%t/err.pcm \
// RUN:   -fsyntax-only -verify
//
// Test header-defined class/alias templates whose deduction guides are
// synthesized in module purview of multiple modules.
// RUN: %clang_cc1 -std=c++20 %t/mod_a.cppm -I%t -emit-module-interface -o %t/mod_a.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod_b.cppm -I%t -emit-module-interface -o %t/mod_b.pcm
// RUN: %clang_cc1 -std=c++20 %t/use_ab.cpp -I%t -fmodule-file=mod_a=%t/mod_a.pcm \
// RUN:   -fmodule-file=mod_b=%t/mod_b.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/use_hdr_and_mod.cpp -I%t \
// RUN:   -fmodule-file=mod_a=%t/mod_a.pcm -fsyntax-only -verify

//--- box.hpp
#ifndef BOX_HPP
#define BOX_HPP

template <class T>
struct Box {
  template <class U>
  Box(U) {}
  Box(T, T) {}
};

template <class T>
Box(T) -> Box<T>;

template <class T>
struct ImplicitBox {
  ImplicitBox(T) {}
};

template <class T>
struct NonTemplateGuideBox {
  NonTemplateGuideBox(T) {}
};

NonTemplateGuideBox(int) -> NonTemplateGuideBox<int>;

template <class T>
struct ConstrainedBox {
  template <class U>
  ConstrainedBox(U) requires (sizeof(U) > 0) {}
};

template <class T>
requires (sizeof(T) > 0)
ConstrainedBox(T) -> ConstrainedBox<T>;

#endif

//--- err.cppm
module;

#include "box.hpp"

export module err;

export template <class T>
using Err = Box<T>;

export template <class T>
using ImplicitErr = ImplicitBox<T>;

export template <class T>
using NonTemplateErr = NonTemplateGuideBox<T>;

export template <class T>
using ConstrainedErr = ConstrainedBox<T>;

auto seed1 = Err(0);
auto seed2 = ImplicitErr(0);
auto seed3 = NonTemplateErr(0);
auto seed4 = ConstrainedErr(0);

//--- use.cpp
// expected-no-diagnostics
#include "box.hpp"

auto pre_implicit = ImplicitBox(1);

import err;

auto result1 = Err(42);
auto result2 = ImplicitErr(42);
auto result3a = NonTemplateErr(42);
auto result3b = NonTemplateErr(3.14);
auto result4 = ConstrainedErr(42);

//--- use_extra.cpp
// expected-no-diagnostics
#include "box.hpp"
import err;

template <class T>
Box(T, T) -> Box<T>;

auto result1 = Err(42);
auto result2 = Err(1, 2);

//--- shared.hpp
#ifndef SHARED_HPP
#define SHARED_HPP

template <class T>
struct SharedBox {
  SharedBox(T) {}
};

template <class T>
struct SharedGuidedBox {
  template <class U>
  SharedGuidedBox(U) {}
};

template <class T>
SharedGuidedBox(T) -> SharedGuidedBox<T>;

template <class T>
using SharedErr = SharedGuidedBox<T>;

#endif

//--- mod_a.cppm
module;
#include "shared.hpp"
export module mod_a;
export using ::SharedBox;
export using ::SharedErr;

auto a1 = SharedBox(0);
auto a2 = SharedErr(0);

//--- mod_b.cppm
module;
#include "shared.hpp"
export module mod_b;

auto b1 = SharedBox(0);
auto b2 = SharedErr(0);

//--- use_ab.cpp
// expected-no-diagnostics
import mod_a;
import mod_b;

auto r1 = SharedBox(42);
auto r2 = SharedErr(42);

//--- use_hdr_and_mod.cpp
// expected-no-diagnostics
#include "shared.hpp"

auto before1 = SharedBox(1);
auto before2 = SharedErr(1);

import mod_a;

auto after1 = SharedBox(42);
auto after2 = SharedErr(42);
