// Test that -print-file-name finds the correct file.

// RUN: %clang -print-file-name=share/asan_ignorelist.txt 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-RESOURCE-DIR %s
// CHECK-RESOURCE-DIR: resource_dir{{/|\\}}share{{/|\\}}asan_ignorelist.txt

// RUN: %clang -print-file-name=libclang_rt.builtins.a 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-COMPILER-RT %s
// CHECK-COMPILER-RT: resource_dir_with_per_target_subdir{{/|\\}}lib{{/|\\}}x86_64-unknown-linux-gnu{{/|\\}}libclang_rt.builtins.a

// RUN: %clang -print-file-name=include/c++/v1 2>&1 \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_libcxx_tree/usr/bin \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-INSTALL-DIR %s
// CHECK-INSTALL-DIR: basic_linux_libcxx_tree{{/|\\}}usr{{/|\\}}bin{{/|\\}}..{{/|\\}}include{{/|\\}}c++{{/|\\}}v1
