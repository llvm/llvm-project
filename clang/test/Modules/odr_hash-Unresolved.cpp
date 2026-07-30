// RUN: rm -rf %t

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/modules.cache \
// RUN:  -I %S/Inputs/odr_hash-Unresolved \
// RUN:  -fmodules \
// RUN:  -fimplicit-module-maps \
// RUN:  -fmodules-cache-path=%t/modules.cache \
// RUN:  -fmodules-local-submodule-visibility \
// RUN:  -std=c++11 -x c++ %s -fsyntax-only

// Note: There is no -verify in the run line because some error messages are
// not captured from the module building stage.

#include "Module2/include.h"
