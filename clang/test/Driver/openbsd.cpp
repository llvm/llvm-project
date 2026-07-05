// Check libraries used when linking C++
// RUN: %clangxx %s -### -o %t.o --target=amd64-pc-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CXX %s
// RUN: %clangxx %s -### -o %t.o --target=i686-pc-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CXX %s
// RUN: %clangxx %s -### -o %t.o --target=aarch64-unknown-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CXX %s
// RUN: %clangxx %s -### -o %t.o --target=arm-unknown-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CXX %s
// CHECK-CXX: "-lc++" "-lc++abi" "-lpthread" "-lm"

// Check for profiling variants of libraries when linking C++
// RUN: %clangxx %s -### -pg -o %t.o --target=amd64-pc-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG-CXX %s
// RUN: %clangxx %s -### -pg -o %t.o --target=i686-pc-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG-CXX %s
// RUN: %clangxx %s -### -pg -o %t.o --target=aarch64-unknown-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG-CXX %s
// RUN: %clangxx %s -### -pg -o %t.o --target=arm-unknown-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG-CXX %s
// CHECK-PG-CXX: "-lc++_p" "-lc++abi_p" "-lpthread_p" "-lm_p"

// Test include paths with a sysroot.
// RUN: %clangxx %s -### -fsyntax-only 2>&1 \
// RUN:     --target=amd64-pc-openbsd \
// RUN:     --sysroot=%S/Inputs/basic_openbsd_libcxx_tree \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-SYSROOT %s
// CHECK-LIBCXX-SYSROOT: "-cc1"
// CHECK-LIBCXX-SYSROOT-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LIBCXX-SYSROOT-SAME: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"

// Test include paths when the sysroot path ends with `/`.
// RUN: %clangxx %s -### -fsyntax-only 2>&1 \
// RUN:     --target=amd64-pc-openbsd \
// RUN:     --sysroot=%S/Inputs/basic_openbsd_libcxx_tree/ \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-SYSROOT-SLASH %s
// CHECK-LIBCXX-SYSROOT-SLASH: "-cc1"
// CHECK-LIBCXX-SYSROOT-SLASH-SAME: "-isysroot" "[[SYSROOT:[^"]+/]]"
// CHECK-LIBCXX-SYSROOT-SLASH-SAME: "-internal-isystem" "[[SYSROOT]]usr/include/c++/v1"
