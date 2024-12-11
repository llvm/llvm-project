
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/ptrcheck-include-from-darwin/ %s -verify
// expected-no-diagnostics

#include "cdefs.h"

#if !__has_ptrcheck
    #error __has_ptrcheck is not defined to 1
#endif
