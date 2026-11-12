
#include "boxed-exprs.h"

const char *glob_str;

void foo() {
  NSString *s = @(glob_str);
}

// RUN: c-index-test -index-file -target x86_64-apple-macosx10.7 %s | FileCheck -check-prefix=CHECK-INDEX %s
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: glob_str | {{.*}} | loc: 7:19

// RUN: c-index-test -cursor-at=%s:7:24 -target x86_64-apple-macosx10.7 %s | FileCheck -check-prefix=CHECK-CURSOR %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -cursor-at=%s:7:24 -target x86_64-apple-macosx10.7 %s | FileCheck -check-prefix=CHECK-CURSOR %s
// CHECK-CURSOR: 7:19 DeclRefExpr=glob_str:4:13 Extent=[7:19 - 7:27] Spelling=glob_str ([7:19 - 7:27])

// RUN: c-index-test -cursor-at=%S/boxed-exprs.h:9:19 -target x86_64-apple-macosx10.7 %s | FileCheck -check-prefix=CHECK-CURSOR2 %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -cursor-at=%S/boxed-exprs.h:9:19 -target x86_64-apple-macosx10.7 %s | FileCheck -check-prefix=CHECK-CURSOR2 %s
// CHECK-CURSOR2: 9:19 DeclRefExpr=cs:8:38 Extent=[9:19 - 9:21] Spelling=cs ([9:19 - 9:21])
