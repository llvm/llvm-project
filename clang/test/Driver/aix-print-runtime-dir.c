// Test output of -print-runtime-dir on AIX

// RUN: %clang -print-runtime-dir --target=powerpc-ibm-aix \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// RUN: %clang -print-runtime-dir --target=powerpc64-ibm-aix \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// PRINT-RUNTIME-DIR: lib{{/|\\}}aix{{$}}
