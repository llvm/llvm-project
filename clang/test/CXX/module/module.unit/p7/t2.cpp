// RUN: %clang_cc1 -std=c++20 %s -verify
// expected-no-diagnostics
module;

#include "Inputs/h2.h"

export module x;

extern "C++" class CPP {};
