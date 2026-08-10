// RUN: %clangxx -target i686-pc-linux-gnu -### -nostdlib++ %s 2> %t
// RUN: FileCheck < %t %s

// We should still have -lm and the C standard libraries, but not -lstdc++.

// CHECK-NOT: -lstdc++
// CHECK-NOT: -lc++
// CHECK: -lm

// Make sure -lstdc++ isn't rewritten to the default stdlib when -nostdlib++ is
// used.
//
// RUN: %clangxx -target i686-pc-linux-gnu -### \
// RUN:     -nostdlib++ -stdlib=libc++ -lstdc++ %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-RESERVED-LIB-REWRITE < %t %s
// CHECK-RESERVED-LIB-REWRITE: -lstdc++
// CHECK-RESERVED-LIB-REWRITE-NOT: -lc++
