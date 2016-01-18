#include "complete-preamble.h"
void f() {
  std::
}

// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:3:8 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {ResultType void}{TypedText wibble}{LeftParen (}{RightParen )} (50)

