// RUN: c-index-test -test-inclusion-stack-source %s 2>&1 | FileCheck %s

#include "include_test.h"

// CHECK: cindex-test-inclusions.c
// CHECK: included by:
// CHECK: include_test.h
// CHECK: included by:
// CHECK: cindex-test-inclusions.c:3:10
// CHECK: include_test_2.h
// CHECK: included by:
// CHECK: include_test.h:1:10
// CHECK: cindex-test-inclusions.c:3:10

// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-inclusion-stack-source %s 2>&1 | FileCheck -check-prefix=REPARSE %s
// REPARSE: include_test_2.h
// REPARSE: included by:
// REPARSE: include_test.h:1:10
// REPARSE: cindex-test-inclusions.c:3:10
// REPARSE: include_test.h
// REPARSE: included by:
// REPARSE: cindex-test-inclusions.c:3:10
// REPARSE: cindex-test-inclusions.c
// REPARSE: included by:
