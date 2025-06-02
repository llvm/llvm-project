// Test output of -print-runtime-dir on AIX

// RUN: %clang -print-runtime-dir --target=powerpc-ibm-aix \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// RUN: %clang -print-runtime-dir --target=powerpc64-ibm-aix \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// RUN: %clang -print-runtime-dir --target=powerpc-ibm-aix \
// RUN:        -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir\
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR32-PER-TARGET %s

// RUN: %clang -print-runtime-dir --target=powerpc64-ibm-aix \
// RUN:        -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir\
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR64-PER-TARGET %s

// RUN: %clang -print-runtime-dir --target=powerpc-ibm-aix-unknown \
// RUN:        -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR32-UNKNOWN-ENV %s

// RUN: %clang -print-runtime-dir --target=powerpc64-ibm-aix-unknown \
// RUN:        -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR64-UNKNOWN-ENV %s 

// PRINT-RUNTIME-DIR: lib{{/|\\}}aix{{$}}
// PRINT-RUNTIME-DIR32-PER-TARGET: lib{{/|\\}}powerpc-ibm-aix{{$}}
// PRINT-RUNTIME-DIR64-PER-TARGET: lib{{/|\\}}powerpc64-ibm-aix{{$}}
// PRINT-RUNTIME-DIR32-UNKNOWN-ENV: lib{{/|\\}}powerpc-ibm-aix
// PRINT-RUNTIME-DIR64-UNKNOWN-ENV: lib{{/|\\}}powerpc64-ibm-aix
