// RUN: %clang_cc1 -std=c++20 %s -verify
// expected-no-diagnostics
module;

#include "Inputs/h4.h"

export module x;

extern "C" struct C {
  int a;
  int b;
  double d;
};
