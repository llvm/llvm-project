// RUN: %clangxx -target i686-pc-linux-gnu -### -nostdlib++ %s 2> %t
// RUN: FileCheck < %t %s

// We should still have -lm and the C standard libraries, but not -lstdc++.

// CHECK-NOT: -lstdc++
// CHECK-NOT: -lc++
// CHECK: -lm
