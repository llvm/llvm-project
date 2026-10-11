// Check that clang emits the "target-abi" module flag for ARM/Thumb using the
// target ABI string.

// Default ABIs (no -target-abi override).
// RUN: %clang_cc1 -triple armv7-linux-gnueabihf -emit-llvm -o - %s | FileCheck --check-prefix=AAPCS-LINUX %s
// RUN: %clang_cc1 -triple armv7-none-eabi -emit-llvm -o - %s | FileCheck --check-prefix=AAPCS %s
// RUN: %clang_cc1 -triple armv7-apple-darwin -emit-llvm -o - %s | FileCheck --check-prefix=APCS-GNU %s
// RUN: %clang_cc1 -triple armv7k-apple-watchos -emit-llvm -o - %s | FileCheck --check-prefix=AAPCS16 %s
// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -emit-llvm -o - %s | FileCheck --check-prefix=AAPCS-LINUX %s

// Explicit -target-abi override differing from the triple default.
// RUN: %clang_cc1 -triple armv7-linux-gnueabihf -target-abi apcs-gnu -emit-llvm -o - %s | FileCheck --check-prefix=APCS-GNU %s

// AAPCS-LINUX: !{i32 1, !"target-abi", !"aapcs-linux"}
// AAPCS: !{i32 1, !"target-abi", !"aapcs"}
// APCS-GNU: !{i32 1, !"target-abi", !"apcs-gnu"}
// AAPCS16: !{i32 1, !"target-abi", !"aapcs16"}

int x;
