// RUN: rm -rf %t
// RUN: split-file %s %t

// Build A without knowing about the module map for B.
//
// This means "b.h" is included textually and embedded into the PCM for module A
// with B_VALUE set to 1.
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -fmodules-strict-context-hash -I %t/a -I %t/b %t/tu.c -E -o -

// Load A with module map for B being known.
// 
// This should make it so that module A turns the header inclusion into a module
// import of B, which gets built with clean preprocessor state, resulting in the
// macro B_VALUE set to 2.
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -fmodules-strict-context-hash -I %t/a -I %t/b %t/tu.c -E -o - \
// RUN:   -fmodule-map-file=%t/b/b.modulemap | FileCheck %s
//
// CHECK: int value = 2;

//--- tu.c
#include "a.h"
int value = B_VALUE;

//--- a/module.modulemap
module A { header "a.h" export * }
//--- a/a.h
#define WITHIN_A
#include "b.h"

//--- b/b.modulemap
module B { header "b.h" }
//--- b/b.h
#ifdef WITHIN_A
#define B_VALUE 1
#else
#define B_VALUE 2
#endif
