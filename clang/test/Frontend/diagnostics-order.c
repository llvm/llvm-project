// Make sure a note stays with its associated command-line argument diagnostic.
// Previously, these diagnostics were grouped by diagnostic level with all
// notes last.
//
// RUN: not %clang_cc1 -O999 -std=bogus %s 2> %t
// RUN: FileCheck < %t %s
//
// CHECK: warning: optimization level '-O999' is not supported
// CHECK-NEXT: error: invalid value 'bogus' in '-std=bogus'
// CHECK-NEXT: note: use {{.*}} for {{.*}} standard
