// RUN: touch %t.o
// RUN: %clang -flto=thin -fenable-matrix %t.o -### --target=powerpc64-ibm-aix-xcoff 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-THINLTO-MATRIX-AIX
// CHECK-THINLTO-MATRIX-AIX: "-bplugin_opt:-enable-matrix"

// RUN: %clang -flto=thin -fenable-matrix %t.o -### --target=x86_64-unknown-linux 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-THINLTO-MATRIX
// CHECK-THINLTO-MATRIX: "-plugin-opt=-enable-matrix"
