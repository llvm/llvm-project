// RUN: c-index-test -test-load-source-usrs local %s 2>&1 | FileCheck %s

// <rdar://problem/8452791> - Crash when generating USRs.
@interface Rdar8452791 () - (void)rdar8452791;

// CHECK: error: cannot find interface declaration for 'Rdar8452791'
// CHECK: missing '@end'
