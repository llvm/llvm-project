// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -verify -std=c++20 -fskip-odr-check-in-gmf -I. -emit-header-unit -xc++-user-header bz1.h
// RUN: %clang_cc1 -verify -std=c++20 -fskip-odr-check-in-gmf -I. -emit-header-unit -xc++-user-header bz2.h
// RUN: %clang_cc1 -verify -std=c++20 -fskip-odr-check-in-gmf -I. -emit-header-unit -xc++-user-header -fmodule-file=bz1.pcm -fmodule-file=bz2.pcm bz.cpp

//--- compare
namespace std {
namespace __detail {

template<typename _Tp>
inline constexpr unsigned __cmp_cat_id = 1;

template<typename... _Ts>
constexpr auto __common_cmp_cat() {
  (__cmp_cat_id<_Ts> | ...);
}

} // namespace __detail
} // namespace std

//--- bz0.h
template <class T>
int operator|(T, T);

//--- bz1.h
#include "bz0.h"
#include <compare>
// expected-no-diagnostics

//--- bz2.h
#include <compare>
// expected-no-diagnostics

//--- bz.cpp
#include <compare>

import "bz1.h"; // expected-warning {{the implementation of header units is in an experimental phase}}
import "bz2.h"; // expected-warning {{the implementation of header units is in an experimental phase}}
