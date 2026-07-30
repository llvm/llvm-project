// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -I%S/Inputs/multiple-include -fmodules-cache-path=%t -fimplicit-module-maps -verify %s -fmodules-local-submodule-visibility
// expected-no-diagnostics
#include "b.h"
int c = get();
