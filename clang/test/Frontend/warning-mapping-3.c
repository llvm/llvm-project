// Check that -Werror and -Wfatal-error interact properly.
//
// Verify mode doesn't work with fatal errors, just use FileCheck here.
//
// RUN: not %clang_cc1 -Wunused-function -Werror -Wfatal-errors %s 2> %t.err
// RUN: FileCheck < %t.err %s
// CHECK: fatal error: unused function
// CHECK: 1 error generated

static void f0(void) {} // expected-fatal {{unused function}}
