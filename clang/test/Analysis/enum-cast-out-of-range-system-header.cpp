// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,optin.core.EnumCastOutOfRange \
// RUN:   -isystem %S/Inputs \
// RUN:   -verify %s

#include "enum-system-header.h"

void test() {
  bad_cast(100);
}

// expected-no-diagnostics
