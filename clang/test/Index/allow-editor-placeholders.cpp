// RUN: c-index-test -test-load-source all %s 2>&1 | FileCheck %s

<#placeholder#>;

// CHECK-NOT: error
