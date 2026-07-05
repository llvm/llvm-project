// RUN: %clangxx -x c++ %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_libcxx_tree/usr/bin \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   | FileCheck --check-prefix=CHECK-PER-TARGET-RUNTIME %s
// CHECK-PER-TARGET-RUNTIME: "-cc1"
// CHECK-PER-TARGET-RUNTIME: "-resource-dir" "[[RESDIR:[^"]*]]"
// CHECK-PER-TARGET-RUNTIME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-PER-TARGET-RUNTIME: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-PER-TARGET-RUNTIME: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-PER-TARGET-RUNTIME: "--sysroot=[[SYSROOT]]"
// CHECK-PER-TARGET-RUNTIME: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-linux-gnu"

// RUN: %clang --target=aarch64-unknown-linux-android21 -print-file-name=libc++.so 2>&1 \
// RUN:     -ccc-install-dir %S/Inputs/basic_android_libcxx_tree/usr/bin \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-ANDROID21 %s
// CHECK-LIBCXX-ANDROID21: ..{{/|\\}}lib{{/|\\}}aarch64-unknown-linux-android{{/|\\}}libc++.so

// RUN: %clang --target=aarch64-unknown-linux-android23 -print-file-name=libc++.so 2>&1 \
// RUN:     -ccc-install-dir %S/Inputs/basic_android_libcxx_tree/usr/bin \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-ANDROID23 %s
// CHECK-LIBCXX-ANDROID23: ..{{/|\\}}lib{{/|\\}}aarch64-unknown-linux-android23{{/|\\}}libc++.so

// RUN: %clang --target=aarch64-unknown-linux-android26 -print-file-name=libc++.so 2>&1 \
// RUN:     -ccc-install-dir %S/Inputs/basic_android_libcxx_tree/usr/bin \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-ANDROID23 %s

// RUN: %clang --target=aarch64-unknown-linux-android29 -print-file-name=libc++.so 2>&1 \
// RUN:     -ccc-install-dir %S/Inputs/basic_android_libcxx_tree/usr/bin \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-ANDROID29 %s
// CHECK-LIBCXX-ANDROID29: ..{{/|\\}}lib{{/|\\}}aarch64-unknown-linux-android29{{/|\\}}libc++.so

// RUN: %clang --target=aarch64-unknown-linux-android31 -print-file-name=libc++.so 2>&1 \
// RUN:     -ccc-install-dir %S/Inputs/basic_android_libcxx_tree/usr/bin \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-ANDROID29 %s

// RUN: %clang --target=aarch64-unknown-linux-android -print-file-name=libc++.so 2>&1 \
// RUN:     -ccc-install-dir %S/Inputs/basic_android_libcxx_tree/usr/bin \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-ANDROID %s
// CHECK-LIBCXX-ANDROID: ..{{/|\\}}lib{{/|\\}}aarch64-unknown-linux-android{{/|\\}}libc++.so

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-X8664 %s
// CHECK-CLANGRT-X8664: lib{{/|\\}}x86_64-unknown-linux-gnu{{/|\\}}libclang_rt.builtins.a

// RUN: %clang -rtlib=compiler-rt -print-file-name=libclang_rt.builtins.a 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=CHECK-FILE-NAME-X8664 %s
// CHECK-FILE-NAME-X8664: lib{{/|\\}}x86_64-unknown-linux-gnu{{/|\\}}libclang_rt.builtins.a

// RUN: %clang -rtlib=compiler-rt -print-file-name=libclang_rt.builtins.a 2>&1 \
// RUN:     --target=aarch64-unknown-linux-android21 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=CHECK-FILE-NAME-ANDROID21 %s
// CHECK-FILE-NAME-ANDROID21: lib{{/|\\}}aarch64-unknown-linux-android{{/|\\}}libclang_rt.builtins.a

// RUN: %clang -rtlib=compiler-rt -print-file-name=libclang_rt.builtins.a 2>&1 \
// RUN:     --target=aarch64-unknown-linux-android23 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=CHECK-FILE-NAME-ANDROID23 %s
// CHECK-FILE-NAME-ANDROID23: lib{{/|\\}}aarch64-unknown-linux-android23{{/|\\}}libclang_rt.builtins.a

// RUN: %clang -rtlib=compiler-rt -print-file-name=libclang_rt.builtins.a 2>&1 \
// RUN:     --target=aarch64-unknown-linux-android26 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=CHECK-FILE-NAME-ANDROID23 %s

// RUN: %clang -rtlib=compiler-rt -print-file-name=libclang_rt.builtins.a 2>&1 \
// RUN:     --target=aarch64-unknown-linux-android29 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=CHECK-FILE-NAME-ANDROID29 %s
// CHECK-FILE-NAME-ANDROID29: lib{{/|\\}}aarch64-unknown-linux-android29{{/|\\}}libclang_rt.builtins.a

// RUN: %clang -rtlib=compiler-rt -print-file-name=libclang_rt.builtins.a 2>&1 \
// RUN:     --target=aarch64-unknown-linux-android31 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=CHECK-FILE-NAME-ANDROID29 %s

// RUN: %clang -rtlib=compiler-rt -print-file-name=libclang_rt.builtins.a 2>&1 \
// RUN:     --target=aarch64-unknown-linux-android \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=CHECK-FILE-NAME-ANDROID %s
// CHECK-FILE-NAME-ANDROID: lib{{/|\\}}aarch64-unknown-linux-android{{/|\\}}libclang_rt.builtins.a
