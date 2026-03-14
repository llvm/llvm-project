#include "ignore-warnings-from-headers.h"

void g(int unusedInMainFile) {}

// RUN: env CINDEXTEST_IGNORE_NONERRORS_FROM_INCLUDED_FILES=1 c-index-test -test-load-source function %s -Wunused-parameter 2>&1 | FileCheck %s
// CHECK-NOT: warning: unused parameter 'unusedInHeader'
// CHECK: warning: unused parameter 'unusedInMainFile'
