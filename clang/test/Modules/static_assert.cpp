// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps \
// RUN:            -I%S/Inputs/static_assert -std=c++1z -verify %s
// expected-no-diagnostics

#include "a.h"

S s;
