// RUN: %clang -### %s -fno-optimize-sibling-calls 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOSC < %t %s
// CHECK-NOSC: "-mdisable-tail-calls"

// RUN: %clang -### -foptimize-sibling-calls %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-OSC < %t %s
// CHECK-OSC-NOT: "-mdisable-tail-calls"

