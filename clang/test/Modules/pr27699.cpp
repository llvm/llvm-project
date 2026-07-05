// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR27699 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR27699/module.modulemap -fmodules-cache-path=%t -fmodules-local-submodule-visibility -I%S/Inputs/PR27699 -verify %s

#include "Subdir/a.h"
#include "Subdir/b.h"

// expected-no-diagnostics

