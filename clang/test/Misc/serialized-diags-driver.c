// Test that the driver correctly combines its own diagnostics with CC1's in the
// serialized diagnostics. To test this, we need to trigger diagnostics from
// both processes, so we compile code that has a warning (with an associated
// note) and then force the driver to crash. We compile stdin so that the crash
// doesn't litter the user's system with preprocessed output.

// RUN: rm -f %t
// RUN: %clang -Wx-typoed-warning -Wall -fsyntax-only --serialize-diagnostics %t.diag %s
// RUN: c-index-test -read-diagnostics %t.diag 2>&1 | FileCheck %s

// CHECK: warning: unknown warning option '-Wx-typoed-warning'
// CHECK-SAME: [-Wunknown-warning-option] []

// CHECK: warning: variable 'voodoo' is uninitialized when used here [-Wuninitialized]
// CHECK: note: initialize the variable 'voodoo' to silence this warning []
// CHECK: Number of diagnostics: 2

void foo(void) {
  int voodoo;
  voodoo = voodoo + 1;
}
