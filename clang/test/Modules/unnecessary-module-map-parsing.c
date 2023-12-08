// This checks that we are not parsing module maps if modules are not enabled.

// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -I %S/Inputs/unnecessary-module-map-parsing -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -I %S/Inputs/unnecessary-module-map-parsing -fsyntax-only %s

// CHECK: error: expected umbrella, header, submodule, or module export

#include "a1.h"
