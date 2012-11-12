int foo() { }
int *bar(float *f) { return f; }

// RUN: c-index-test -test-load-source all %s 2>&1|FileCheck -check-prefix=CHECK-BOTH-WARNINGS %s
// RUN: c-index-test -test-load-source-reparse 5 all %s 2>&1|FileCheck -check-prefix=CHECK-BOTH-WARNINGS %s
// RUN: c-index-test -test-load-source all -Wno-return-type  %s 2>&1|FileCheck -check-prefix=CHECK-SECOND-WARNING %s
// RUN: c-index-test -test-load-source-reparse 5 all -Wno-return-type %s 2>&1|FileCheck -check-prefix=CHECK-SECOND-WARNING %s
// RUN: c-index-test -test-load-source all -w %s 2>&1|not grep warning:
// RUN: c-index-test -test-load-source-reparse 5 all -w %s 2>&1|not grep warning:

// CHECK-BOTH-WARNINGS: warning: control reaches end of non-void function
// CHECK-BOTH-WARNINGS: warning: incompatible pointer types returning 'float *' from a function with result type 'int *'

// CHECK-SECOND-WARNING-NOT:control reaches end of non-void
// CHECK-SECOND-WARNING: warning: incompatible pointer types returning 'float *' from a function with result type 'int *'

