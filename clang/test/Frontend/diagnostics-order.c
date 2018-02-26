// Make sure a note stays with its associated command-line argument diagnostic.
// Previously, these diagnostics were grouped by diagnostic level with all
// notes last.
//
// RUN: not %clang_cc1 -O999 -std=bogus -verify=-foo %s 2> %t
// RUN: FileCheck < %t %s
//
// CHECK:      error: invalid value '-foo' in '-verify='
// CHECK-NEXT: note: -verify prefixes must start with a letter and contain only alphanumeric characters, hyphens, and underscores
// CHECK-NEXT: warning: optimization level '-O999' is not supported
// CHECK-NEXT: error: invalid value 'bogus' in '-std=bogus'
// CHECK-NEXT: note: use {{.*}} for {{.*}} standard
