// RUN: rm -f %t.head.h.pch
// RUN: c-index-test -write-pch %t.head.h.pch %s -Wuninitialized -Werror=unused 2>&1 | FileCheck -check-prefix=HEAD_DIAGS %s
// RUN: c-index-test -test-load-source local %s -include %t.head.h -Wuninitialized -Werror=unused 2>&1 | FileCheck -check-prefix=MAIN_DIAGS %s

// Make sure -Wuninitialized works even though the header had a warn-as-error occurrence.

// HEAD_DIAGS: error: unused variable 'x'
// MAIN_DIAGS: warning: variable 'x1' is uninitialized
// MAIN_DIAGS-NOT: error: use of undeclared identifier

#ifndef HEADER
#define HEADER

static void foo_head() {
  int x;
}

#else

void test() {
  int x1; // expected-note {{initialize}}
  int x2 = x1; // expected-warning {{uninitialized}}
  (void)x2;
  foo_head();
}

#endif
