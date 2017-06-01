// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fimplicit-module-maps -fmodules-cache-path=%t.a -I %S/Inputs/preprocess -verify %s
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fimplicit-module-maps -fmodules-cache-path=%t.b -I %S/Inputs/preprocess -x c -verify -x c %s

// expected-no-diagnostics

#include "file.h"

