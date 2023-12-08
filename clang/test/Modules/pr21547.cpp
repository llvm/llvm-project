// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/PR21547 -verify %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/PR21547 -verify %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/PR21547 -emit-llvm-only %s

#include "Inputs/PR21547/FirstHeader.h"

//expected-no-diagnostics
