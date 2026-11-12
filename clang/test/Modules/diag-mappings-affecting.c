// Test that pruning non-affecting input files happens before serializing
// diagnostic pragma mappings.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I %t/include_a -I %t/include_textual -fsyntax-only %t/tu.c

//--- tu.c
#include "a1.h"

//--- include_a/module.modulemap
module A {
  header "a1.h"
  header "a2.h"
}
//--- include_a/a1.h
#include "textual.h" // This will also load the non-affecting
                     // include_textual/module.modulemap.
#include "a2.h"
//--- include_a/a2.h
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic pop

//--- include_textual/module.modulemap
//--- include_textual/textual.h
