// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/merge-dependent-friends -verify %s
// expected-no-diagnostics
#include "d.h"
