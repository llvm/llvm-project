// Test that -print-search-dirs includes system library paths on AIX

// RUN: %clang -print-search-dirs --target=powerpc-ibm-aix7.3.0.0 \
// RUN:        --sysroot=%S/Inputs/aix_ppc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PRINT-SEARCH-DIRS-32 %s

// CHECK-PRINT-SEARCH-DIRS-32: programs: =
// CHECK-PRINT-SEARCH-DIRS-32: libraries: =
// CHECK-PRINT-SEARCH-DIRS-32-SAME: {{.*}}/usr/lib
// CHECK-PRINT-SEARCH-DIRS-32-SAME: {{.*}}/lib

// RUN: %clang -print-search-dirs --target=powerpc64-ibm-aix7.3.0.0 \
// RUN:        --sysroot=%S/Inputs/aix_ppc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PRINT-SEARCH-DIRS-64 %s

// CHECK-PRINT-SEARCH-DIRS-64: programs: =
// CHECK-PRINT-SEARCH-DIRS-64: libraries: =
// CHECK-PRINT-SEARCH-DIRS-64-SAME: {{.*}}/usr/lib
// CHECK-PRINT-SEARCH-DIRS-64-SAME: {{.*}}/lib
