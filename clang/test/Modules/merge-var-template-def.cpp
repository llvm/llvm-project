// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/merge-var-template-def -std=c++11 -verify %s
// RUN: %clang_cc1 -I%S/Inputs/merge-var-template-def -std=c++11 -verify -fmodules -Werror=undefined-internal -fmodules-local-submodule-visibility -fmodules-cache-path=%t -fimplicit-module-maps %s
// expected-no-diagnostics

#include "b2.h"
const bool *y = &S<bool, false>::value;
