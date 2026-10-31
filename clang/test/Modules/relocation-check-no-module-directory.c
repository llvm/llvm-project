// Check that a transitively imported module can still be resolved when the PCM
// has no MODULE_DIRECTORY record, which -fmodule-file-home-is-cwd suppresses.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Case 1: the load of the transitively imported 'Other' should succeed.
// RUN: %clang -fmodules -fimplicit-module-maps -fsyntax-only %t/tu.c \
// RUN:   -fmodules-cache-path=%t/cache -I%t/pathB -I%t/pathC \
// RUN:   -Xclang -fmodule-file-home-is-cwd

// Case 2: the same load must not crash when module validation is disabled.
// RUN: %clang -fmodules -fimplicit-module-maps -fsyntax-only %t/tu.c \
// RUN:   -fmodules-cache-path=%t/cache-novalidate -I%t/pathB -I%t/pathC \
// RUN:   -Xclang -fmodule-file-home-is-cwd -Xclang -fno-validate-pch

//--- pathB/module.modulemap
module Dep { header "Dep.h" export * }
//--- pathB/Dep.h
#include "Other.h"
int dep(void);

//--- pathC/module.modulemap
module Other { header "Other.h" }
//--- pathC/Other.h
int other(void);

//--- tu.c
#include "Dep.h"
