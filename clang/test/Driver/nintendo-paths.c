// REQUIRES: symlinks, aarch64-registered-target

/// Check default c include paths.
// RUN: %clang -### --target=aarch64-nintendo-nn %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK %s

/// Check default c++ include paths.
// RUN: %clang -### --target=aarch64-nintendo-nn -x c++ %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,CXX %s

/// Check default kernel c include paths.
// RUN: %clang -### --target=aarch64-nintendo-nn -mkernel %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,KERNEL %s

/// Check default kernel c++ include paths.
// RUN: %clang -### --target=aarch64-nintendo-nn -x c++ -mkernel %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,KERNEL,KERNEL-CXX %s

// CXX: "-internal-isystem" "{{.*}}include{{/|\\\\}}aarch64-nintendo-nn{{/|\\\\}}c++{{/|\\\\}}v1"
// CXX-SAME: "-internal-isystem" "{{.*}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// KERNEL-CXX: "-internal-isystem" "{{.*}}include{{/|\\\\}}kernel{{/|\\\\}}c++{{/|\\\\}}v1"
// KERNEL: "-internal-isystem" "{{.*}}include{{/|\\\\}}kernel"
// CHECK: "-internal-isystem" "{{.*}}include{{/|\\\\}}aarch64-nintendo-nn"
// CHECK-SAME: "-internal-isystem" "{{.*}}include"

/// Check use of explicit sysroot.
// RUN: rm -rf %S/Inputs/nintendo_nn_tree/lib && mkdir -p %S/Inputs/nintendo_nn_tree/lib
// RUN: %clang -### --target=aarch64-nintendo-nn --sysroot=%S/Inputs/nintendo_nn_tree %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=SYSROOT %s
// SYSROOT: "-isysroot" "[[SYSROOT:[^"]+]]"
// SYSROOT-SAME: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}include{{/|\\\\}}aarch64-nintendo-nn"
// SYSROOT-SAME: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}include"

/// Check -nostdinc.
// RUN: %clang -### --target=aarch64-nintendo-nn -nostdinc %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOSTD %s
// NOSTD: "-resource-dir"
// NOSTD-NOT: "-internal-isystem" "{{.*}}include{{/|\\\\}}aarch64-nintendo-nn"

// RUN: ln -sf %clang %S/Inputs/nintendo_nn_tree/bin/clang
// RUN: rm -rf %S/Inputs/nintendo_nn_tree/lib

/// Check warning of incomplete layout (missing lib).
// RUN: %S/Inputs/nintendo_nn_tree/bin/clang -### --target=aarch64-nintendo-nn -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=WARN %s
// WARN: warning: invalid Nintendo toolchain layout
