// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR27513 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR27513/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR27513 -verify %s

#include "Inputs/PR27513/a.h"

//expected-no-diagnostics
