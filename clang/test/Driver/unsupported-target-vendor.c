// Tests that clang does not crash with invalid vendors in target triples.
//
// RUN: %clang --target=arm-apple-firmware -### %s 2>&1 | FileCheck -check-prefix CHECK_APPLE %s
// RUN: %clang_cl --target=arm-apple-firmware -### %s 2>&1 | FileCheck -check-prefix CHECK_APPLE %s

// CHECK_APPLE-NOT: LLVM ERROR: the firmware target os is only supported for the apple vendor


// RUN: not %clang --target=arm-none-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-none-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-unknown-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-unknown-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-pc-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-pc-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-scei-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-scei-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-sie-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-sie-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-fsl-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-fsl-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-ibm-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-ibm-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-img-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-img-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-mti-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-mti-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-nvidia-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-nvidia-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-csr-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-csr-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-amd-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-amd-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-mesa-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-mesa-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-suse-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-suse-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-oe-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-oe-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-intel-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-intel-firmware -### %s 2>&1 | FileCheck %s

// RUN: not %clang --target=arm-meta-firmware -### %s 2>&1 | FileCheck %s
// RUN: not %clang_cl --target=arm-meta-firmware -### %s 2>&1 | FileCheck %s

// CHECK: LLVM ERROR: the firmware target os is only supported for the apple vendor
