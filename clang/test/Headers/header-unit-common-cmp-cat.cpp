// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -verify -std=c++20 -fskip-odr-check-in-gmf -emit-header-unit -xc++-user-header bz0.h
// RUN: %clang_cc1 -verify -std=c++20 -fskip-odr-check-in-gmf -emit-header-unit -xc++-user-header bz1.h
// RUN: %clang_cc1 -verify -std=c++20 -fskip-odr-check-in-gmf -emit-header-unit -xc++-user-header -fmodule-file=bz0.pcm -fmodule-file=bz1.pcm bz.cpp

//--- compare
template<typename _Tp>
inline constexpr unsigned __cmp_cat_id = 1;

template<typename... _Ts>
constexpr auto __common_cmp_cat() {
  (__cmp_cat_id<_Ts> | ...);
}

//--- bz0.h
template <class T>
int operator|(T, T);

#include "compare"
// expected-no-diagnostics

//--- bz1.h
#include "compare"
// expected-no-diagnostics

//--- bz.cpp
#include "compare"

import "bz0.h"; // expected-warning {{the implementation of header units is in an experimental phase}}
import "bz1.h"; // expected-warning {{the implementation of header units is in an experimental phase}}
