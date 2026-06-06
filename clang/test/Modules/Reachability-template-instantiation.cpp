// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Templ.cppm -emit-module-interface -o %t/Templ.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cppm -fprebuilt-module-path=%t -emit-module-interface -o %t/Use.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use.cpp -verify -fsyntax-only

// RUN: %clang_cc1 -std=c++20 %t/Templ.cppm -emit-reduced-module-interface -o %t/Templ.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cppm -fprebuilt-module-path=%t -emit-reduced-module-interface -o %t/Use.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use.cpp -verify -fsyntax-only
//
//--- Templ.h
#ifndef TEMPL_H
#define TEMPL_H
template <class T>
class Wrapper {
public:
  T value;
};
#endif

//--- Templ.cppm
export module Templ;
export template <class T>
class Wrapper2 {
public:
  T value;
};

//--- Use.cppm
module;
#include "Templ.h"
export module Use;
import Templ;

export template <class T>
class Use {
public:
  Wrapper<T> value;
  Wrapper2<T> value2;
};

export template <class T>
Wrapper<T> wrapper;

//--- Use.cpp
// expected-no-diagnostics
module;
#include "Templ.h"
export module User;

export template <class T>
class User {
public:
  Wrapper<T> value;
};
