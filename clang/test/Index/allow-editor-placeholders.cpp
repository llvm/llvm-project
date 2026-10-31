// RUN: c-index-test -test-load-source all %s 2>&1 | FileCheck %s
// RUN: c-index-test -test-load-source all --driver-mode=cl --target=x86_64-pc-windows-msvc /WX %s 2>&1 | FileCheck %s

<#placeholder#>;

// CHECK-NOT: error:
