#include "truncation.c.h"

/* foo */

struct 

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s.h:4:8 -o - %s | FileCheck -check-prefix=CC1 %s
// CHECK-CC1: X
// CHECK-CC1-NEXT: Y
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:5:8 -o - %s | FileCheck -check-prefix=CC2 %s
// CHECK-CC2: X
// CHECK-CC2: Xa
// CHECK-CC2: Y

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:3:3 -o - %s
