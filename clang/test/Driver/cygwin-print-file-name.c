// Test for compiler-rt library path for cygwin
//
// RUN: %clang --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        --target=x86_64-pc-windows-cygnus \
// RUN:        -print-file-name=libclang_rt.builtins-x86_64.a \
// RUN:      | FileCheck --check-prefix=CHECK %s
// CHECK: {{.*}}{{/|\\}}lib{{/|\\}}cygwin{{/|\\}}libclang_rt.builtins-x86_64.a
//
// RUN: %clang --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:        -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:        --target=x86_64-pc-windows-cygnus \
// RUN:        -print-file-name=libclang_rt.builtins.a \
// RUN:      | FileCheck --check-prefix=CHECK_PER_TARGET %s
// CHECK_PER_TARGET: {{.*}}{{/|\\}}lib{{/|\\}}x86_64-pc-windows-cygnus{{/|\\}}libclang_rt.builtins.a
