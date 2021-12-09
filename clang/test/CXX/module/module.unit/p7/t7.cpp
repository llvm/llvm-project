// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: %clang_cc1 -std=c++20 -I%S/Inputs/ %s -verify
// expected-no-diagnostics
module;
#include "h7.h"
export module X;
