// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/PR24954 -verify %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/PR24954 -verify %s

#include "A.h"

// expected-no-diagnostics
