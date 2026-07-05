// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++1z -I%S/Inputs/odr_hash-elaborated-types -verify %s
// RUN: %clang_cc1 -std=c++1z -fmodules -fmodules-local-submodule-visibility -fmodule-map-file=%S/Inputs/odr_hash-elaborated-types/module.modulemap -fmodules-cache-path=%t -x c++ -I%S/Inputs/odr_hash-elaborated-types -verify %s

#include "textual_stat.h"

#include "first.h"
#include "second.h"

void use() { struct stat value; }

// expected-no-diagnostics
