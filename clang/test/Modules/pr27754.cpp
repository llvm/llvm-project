// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR27754 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR27754/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR27754/ -verify %s

#include "TMetaUtils.h"

// expected-no-diagnostics
