// General tests that the correct versions of values-*.o are used on
// Solaris targets sane. Note that we use sysroot to make these tests
// independent of the host system.

// Check sparc-sun-solaris2.11, 32bit
// RUN: %clang -ansi -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-ANSI %s
// CHECK-LD-SPARC32-ANSI: values-Xc.o
// CHECK-LD-SPARC32-ANSI: values-xpg6.o

// RUN: %clang -std=c89 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-C89 %s
// CHECK-LD-SPARC32-C89: values-Xc.o
// CHECK-LD-SPARC32-C89: values-xpg4.o

// RUN: %clang -std=c90 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-C90 %s
// CHECK-LD-SPARC32-C90: values-Xc.o
// CHECK-LD-SPARC32-C90: values-xpg4.o

// RUN: %clang -std=iso9899:199409 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-C94 %s
// CHECK-LD-SPARC32-C94: values-Xc.o
// CHECK-LD-SPARC32-C94: values-xpg4.o

// RUN: %clang -std=c11 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-C11 %s
// CHECK-LD-SPARC32-C11: values-Xc.o
// CHECK-LD-SPARC32-C11: values-xpg6.o

// RUN: %clang -std=gnu89 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-GNU89 %s
// CHECK-LD-SPARC32-GNU89: values-Xa.o
// CHECK-LD-SPARC32-GNU89: values-xpg4.o

// RUN: %clang -std=gnu90 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-GNU90 %s
// CHECK-LD-SPARC32-GNU90: values-Xa.o
// CHECK-LD-SPARC32-GNU90: values-xpg4.o

// RUN: %clang -std=gnu11 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-GNU11 %s
// CHECK-LD-SPARC32-GNU11: values-Xa.o
// CHECK-LD-SPARC32-GNU11: values-xpg6.o

// Check i386-pc-solaris2.11, 32bit
// RUN: %clang -ansi -### %s 2>&1 \
// RUN:     --target=i386-pc-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X32-ANSI %s
// CHECK-LD-X32-ANSI: values-Xc.o
// CHECK-LD-X32-ANSI: values-xpg6.o
