// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -verify -std=c++20 -emit-header-unit -xc++-user-header bz0.h
// RUN: %clang_cc1 -verify -std=c++20 -emit-header-unit -xc++-user-header -fmodule-file=bz0.pcm bz.cpp

//--- compare
#pragma GCC visibility push(default)
#pragma GCC visibility pop

//--- bz0.h
#include "compare"
// expected-no-diagnostics

//--- bz.cpp
#include "compare"

import "bz0.h"; // expected-warning {{the implementation of header units is in an experimental phase}}
