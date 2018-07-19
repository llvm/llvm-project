// RUN: %clang -no-canonical-prefixes -target x86_64-unknown-ananas -static %s \
// RUN:   --sysroot=%S/Inputs/ananas-tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC: ld{{.*}}" "-Bstatic"
// CHECK-STATIC: crt0.o
// CHECK-STATIC: crti.o
// CHECK-STATIC: crtbegin.o
// CHECK-STATIC: crtend.o
// CHECK-STATIC: crtn.o
