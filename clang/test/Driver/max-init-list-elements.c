// RUN: %clang -### -fsyntax-only -fmax-init-list-elements=4 %s 2>&1 | \
// RUN:   FileCheck %s

// CHECK: "-cc1"
// CHECK-SAME: "-fmax-init-list-elements=4"
