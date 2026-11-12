// Test that -print-search-dirs includes system library paths on AIX

// RUN: %clang -print-search-dirs --target=powerpc-ibm-aix7.3.0.0 \
// RUN:        --sysroot=%S/Inputs/aix_ppc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PRINT-SEARCH-DIRS %s

// RUN: %clang -print-search-dirs --target=powerpc64-ibm-aix7.3.0.0 \
// RUN:        --sysroot=%S/Inputs/aix_ppc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PRINT-SEARCH-DIRS %s

// CHECK-PRINT-SEARCH-DIRS: programs: =
// CHECK-PRINT-SEARCH-DIRS: libraries: =
// CHECK-PRINT-SEARCH-DIRS-SAME: {{.*}}/usr/lib
// CHECK-PRINT-SEARCH-DIRS-SAME: {{.*}}/lib
