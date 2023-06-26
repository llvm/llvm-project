// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm64e-apple-ios                              < /dev/null | FileCheck %s --check-prefix=NONE
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm64e-apple-ios -fptrauth-abi-version=0      < /dev/null | FileCheck %s --check-prefix=ZERO
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm64e-apple-ios -fptrauth-kernel-abi-version < /dev/null | FileCheck %s --check-prefix=ZERO
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm64e-apple-ios -fptrauth-abi-version=5      < /dev/null | FileCheck %s --check-prefix=FIVE

// ZERO: #define __ptrauth_abi_version__ 0
// FIVE: #define __ptrauth_abi_version__ 5
// NONE-NOT: __ptrauth_abi_version__
