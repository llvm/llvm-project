// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/redecl-found-building-chains -verify %s
// expected-no-diagnostics
int n, m;
#include "d.h"
A a;
