int
renamable = 0;

// RUN: clang-refactor-test list-actions -at=%s:2:1 %s | FileCheck --check-prefix=CHECK-RENAME %s

// CHECK-RENAME: Found {{[0-9]*}} actions:
// CHECK-RENAME-NEXT: Rename

// RUN: not clang-refactor-test list-actions -at=%s:2:13 %s 2>&1 | FileCheck --check-prefix=CHECK-NONE %s

// CHECK-NONE: No refactoring actions are available at the given location
// CHECK-NONE-NOT: Rename

// RUN: not clang-refactor-test list-actions -at=%s %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
// CHECK-ERR: error: The -at option must use the <file:line:column> format

void localVsGlobalRename(int renamable) { }

// RUN: clang-refactor-test list-actions -dump-raw-action-type -at=%s:17:30 %s | FileCheck --check-prefix=CHECK-LOCAL-RENAME %s

// CHECK-LOCAL-RENAME: Found {{[0-9]*}} actions:
// CHECK-LOCAL-RENAME-NEXT: Rename(1)

namespace nullDeclNamespace {

template<template<typename T> class C> class NullNode {};

struct AfterNull { };
// RUN: clang-refactor-test list-actions -at=%s:28:8 %s | FileCheck --check-prefix=CHECK-RENAME %s

}

#define MACRO(X) (void)X;
void macroArg() {
  int variable = 0;
  MACRO(variable);
}
// RUN: not clang-refactor-test list-actions -at=%s:26:9 -selected=%s:36:9-36:16 %s 2>&1 | FileCheck --check-prefix=CHECK-MACRO-ARG %s
// CHECK-MACRO-ARG: No refactoring actions are available at the given location
