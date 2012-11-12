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
