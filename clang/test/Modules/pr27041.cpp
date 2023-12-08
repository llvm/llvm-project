// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR27041 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR27041/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR27041 -verify %s

#include "Rtypes.h"

// expected-no-diagnostics
