// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodules-local-submodule-visibility \
// RUN:            -fmodule-map-file=%S/Inputs/anon-redecl/module.modulemap \
// RUN:            -I%S/Inputs/anon-redecl \
// RUN:            -verify -std=c++11 %s

#include "a.h"
#include "b.h"
#include "c1.h"
#include "c2.h"

// expected-no-diagnostics
int x = a({});
int y = b({});
int z = c({});
