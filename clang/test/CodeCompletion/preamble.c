#include "some_struct.h"
void foo() {
  struct X x;
  x.

// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:%(line-2):5 -Xclang -code-completion-patterns  %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: FieldDecl:{ResultType int}{TypedText m} (35)
