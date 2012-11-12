// Check that the name of the arch we bind is "ppc" not "powerpc".
//
// RUN: %clang -target powerpc-apple-darwin8 -### \
// RUN:   -ccc-print-phases %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-POWERPC < %t %s
//
// CHECK-POWERPC: bind-arch, "ppc"
