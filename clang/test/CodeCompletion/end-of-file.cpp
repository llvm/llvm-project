// Check that clang does not crash when completing at the last char in the
// buffer.
// NOTE: This file must *NOT* have newline at the end.
// RUN: %clang_cc1 -code-completion-at=%s:%(line+3):2 %s | FileCheck %s
// CHECK: COMPLETION: foo
using foo = int***;
f