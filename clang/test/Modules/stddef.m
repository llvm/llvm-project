@import StdDef.Other;

size_t getSize(void);

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fbuiltin-headers-in-system-modules -fmodules-cache-path=%t -I %S/Inputs/StdDef %s -verify
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/StdDef %s -verify
// expected-no-diagnostics
