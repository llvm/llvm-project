// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/pr19692 -verify %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/pr19692 -verify %s
#include "TFoo.h"
#include "stdint.h"

int k = INT64_MAX; // expected-no-diagnostics
