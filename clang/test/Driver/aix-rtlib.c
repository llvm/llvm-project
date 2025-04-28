// Check the default rtlib for AIX.
// RUN: %clang --target=powerpc-ibm-aix -print-libgcc-file-name \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck -check-prefix=CHECK32 %s
// RUN: %clang --target=powerpc64-ibm-aix -print-libgcc-file-name \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck -check-prefix=CHECK64 %s

// CHECK32: resource_dir{{/|\\}}lib{{/|\\}}aix{{/|\\}}libclang_rt.builtins-powerpc.a
// CHECK64: resource_dir{{/|\\}}lib{{/|\\}}aix{{/|\\}}libclang_rt.builtins-powerpc64.a
