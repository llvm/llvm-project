// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR25501/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR25501 -verify %s

#include "a2.h"
#include "b.h"

auto use = aaa;

// expected-no-diagnostics
