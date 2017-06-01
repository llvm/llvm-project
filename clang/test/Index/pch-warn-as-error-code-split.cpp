// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source local %s -Wuninitialized -Werror=unused 2>&1 | FileCheck -check-prefix=DIAGS %s

// Make sure -Wuninitialized works even though the header had a warn-as-error occurrence.

// DIAGS: error: unused variable 'x'
// DIAGS: warning: variable 'x1' is uninitialized
// DIAGS-NOT: error: use of undeclared identifier
// DIAGS: warning: variable 'x1' is uninitialized

#include "pch-warn-as-error-code-split.h"

void test() {
  int x1; // expected-note {{initialize}}
  int x2 = x1; // expected-warning {{uninitialized}}
  (void)x2;
  foo_head();
}
