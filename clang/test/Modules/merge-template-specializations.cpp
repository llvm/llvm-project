// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -fmodules-local-submodule-visibility -I%S/Inputs/merge-template-specializations -std=c++11 -verify %s
// expected-no-diagnostics
#include "c.h"
X x;
