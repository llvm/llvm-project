// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR28332 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR28332/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR28332 -verify %s

#include "a.h"

// expected-no-diagnostics

