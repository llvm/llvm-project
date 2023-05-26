// Check powerpc64-unknown-linux-gnu. -K not supported.
// RUN: %clang %s 2>&1 -### \
// RUN:        --target=powerpc64-unknown-linux-gnu \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:        -K \
// RUN:   | FileCheck --check-prefixes=CHECK-K-SUPPORT %s
// CHECK-K-SUPPORT: clang: error: unsupported option '-K' for target 'powerpc64-unknown-linux-gnu'
