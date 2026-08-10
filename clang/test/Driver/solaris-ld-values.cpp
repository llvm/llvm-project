// General tests that the correct versions of values-*.o are used on
// Solaris targets sane. Note that we use sysroot to make these tests
// independent of the host system.

// Check sparc-sun-solaris2.11, 32bit
// RUN: %clang -ansi -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-ANSI %s
// CHECK-LD-SPARC32-ANSI: values-Xc.o
// CHECK-LD-SPARC32-ANSI: values-xpg6.o

// RUN: %clang -std=c++98 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-CPP98 %s
// CHECK-LD-SPARC32-CPP98: values-Xc.o
// CHECK-LD-SPARC32-CPP98: values-xpg6.o

// RUN: %clang -std=c++11 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-CPP11 %s
// CHECK-LD-SPARC32-CPP11: values-Xc.o
// CHECK-LD-SPARC32-CPP11: values-xpg6.o

// RUN: %clang -std=gnu++98 -### %s 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32-GNUPP98 %s
// CHECK-LD-SPARC32-GNUPP98: values-Xa.o
// CHECK-LD-SPARC32-GNUPP98: values-xpg6.o

// Check i386-pc-solaris2.11, 32bit
// RUN: %clang -ANSI -### %s 2>&1 \
// RUN:     --target=i386-pc-solaris2.11 \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X32-ANSI %s
// CHECK-LD-X32-ANSI: values-Xa.o
// CHECK-LD-X32-ANSI: values-xpg6.o
