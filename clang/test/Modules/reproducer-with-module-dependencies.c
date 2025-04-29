// Test generating a reproducer for a modular build where required modules are
// built explicitly as separate steps.

// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: c-index-test core -gen-deps-reproducer -working-dir %t \
// RUN:   -- clang-executable -c %t/reproducer.c -o %t/reproducer.o \
// RUN:      -fmodules -fmodules-cache-path=%t | FileCheck %t/reproducer.c

// Test a failed attempt at generating a reproducer.
// RUN: not c-index-test core -gen-deps-reproducer -working-dir %t \
// RUN:   -- clang-executable -c %t/failed-reproducer.c -o %t/reproducer.o \
// RUN:      -fmodules -fmodules-cache-path=%t 2>&1 | FileCheck %t/failed-reproducer.c

//--- modular-header.h
void fn_in_modular_header(void);

//--- module.modulemap
module Test { header "modular-header.h" export * }

//--- reproducer.c
// CHECK: Sources and associated run script(s) are located at:
#include "modular-header.h"

void test(void) {
  fn_in_modular_header();
}

//--- failed-reproducer.c
// CHECK: fatal error: 'non-existing-header.h' file not found
#include "non-existing-header.h"
