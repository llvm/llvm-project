// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core,nullability,apiModeling -verify %s

#include "Inputs/system-header-simulator-for-nullability-cxx.h"

// expected-no-diagnostics

void blah() {
  foo(); // no-crash
}
