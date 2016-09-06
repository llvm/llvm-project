// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/PR26014 -verify %s
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%S/Inputs/PR26014/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR26014 -verify %s

#include "A.h"

// expected-no-diagnostics
