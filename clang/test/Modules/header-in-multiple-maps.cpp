// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/header-in-multiple-maps -fmodule-map-file=%S/Inputs/header-in-multiple-maps/map1 -verify %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/header-in-multiple-maps -fmodule-map-file=%S/Inputs/header-in-multiple-maps/map2 -verify %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/header-in-multiple-maps -fmodule-map-file=%S/Inputs/header-in-multiple-maps/map3 -verify %s
// expected-no-diagnostics

#include "a.h"
#include "a.h"
A *p;
