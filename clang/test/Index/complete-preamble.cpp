#include "complete-preamble.h"
void f() {
  std::
}

// RUN: env CINDEXTEST_EDITING=1 LIBCLANG_TIMING=1 c-index-test -code-completion-at=%s:3:8 %s -o - 2>&1 | FileCheck -check-prefix=CHECK-CC1 -check-prefix=SECOND %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_CREATE_PREAMBLE_ON_FIRST_PARSE=1 LIBCLANG_TIMING=1 c-index-test -code-completion-at=%s:3:8 %s -o - 2>&1 | FileCheck -check-prefix=CHECK-CC1 -check-prefix=FIRST %s

// FIRST: Precompiling preamble
// FIRST: Parsing
// FIRST: Reparsing

// SECOND: Parsing
// SECOND: Precompiling preamble
// SECOND: Reparsing

// CHECK-CC1: {ResultType void}{TypedText wibble}{LeftParen (}{RightParen )} (50)
