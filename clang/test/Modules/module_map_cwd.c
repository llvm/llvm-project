// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'module X { header "x.h" }' > %t/map
// RUN: echo 'extern int n;' > %t/x.h
// RUN: cd %t
// RUN: %clang_cc1 %s -fmodules -fmodule-map-file=map -fmodules-cache-path=. -verify -I.
// expected-no-diagnostics
#include "x.h"
int *m = &n;
