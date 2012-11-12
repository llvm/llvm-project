// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-note-include-stack %s 2>&1 | FileCheck %s -check-prefix=STACK
// RUN: %clang_cc1 -fsyntax-only -fno-diagnostics-show-note-include-stack %s 2>&1 | FileCheck %s -check-prefix=STACKLESS
// RUN: %clang_cc1 -fsyntax-only -fno-diagnostics-show-note-include-stack -fdiagnostics-show-note-include-stack %s 2>&1 | FileCheck %s -check-prefix=STACK
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-note-include-stack -fno-diagnostics-show-note-include-stack %s 2>&1 | FileCheck %s -check-prefix=STACKLESS
// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s -check-prefix=STACKLESS

#include "Inputs/include.h"
int test() {
  return foo(1, 1);
}

bool macro(int x, int y) {
  return EQUALS(&x, y);
}

// STACK: error: no matching function for call to 'foo'
// STACK:  In file included from
// STACK: note: candidate function not viable
// STACK: error: comparison between pointer and integer
// STACK:  In file included from
// STACK: note: expanded from macro

// STACKLESS: error: no matching function for call to 'foo'
// STACKLESS-NOT:  In file included from
// STACKLESS: note: candidate function not viable
// STACKLESS: error: comparison between pointer and integer
// STACKLESS-NOT:  In file included from
// STACKLESS: note: expanded from macro
