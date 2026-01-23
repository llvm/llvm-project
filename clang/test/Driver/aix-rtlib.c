// Check the default rtlib for AIX.
// RUN: %clang --target=powerpc-ibm-aix -print-libgcc-file-name \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck -check-prefix=CHECK32 %s
// RUN: %clang --target=powerpc64-ibm-aix -print-libgcc-file-name \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck -check-prefix=CHECK64 %s
// RUN: %clang --target=powerpc-ibm-aix -print-libgcc-file-name \
// RUN:        -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck -check-prefix=CHECK32-PER-TARGET %s
// RUN: %clang --target=powerpc64-ibm-aix -print-libgcc-file-name \
// RUN:        -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck -check-prefix=CHECK64-PER-TARGET %s

// CHECK32: resource_dir{{/|\\}}lib{{/|\\}}aix{{/|\\}}libclang_rt.builtins-powerpc.a
// CHECK64: resource_dir{{/|\\}}lib{{/|\\}}aix{{/|\\}}libclang_rt.builtins-powerpc64.a
// CHECK32-PER-TARGET: resource_dir_with_per_target_subdir{{/|\\}}lib{{/|\\}}powerpc-ibm-aix{{/|\\}}libclang_rt.builtins.a
// CHECK64-PER-TARGET: resource_dir_with_per_target_subdir{{/|\\}}lib{{/|\\}}powerpc64-ibm-aix{{/|\\}}libclang_rt.builtins.a
