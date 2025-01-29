// RUN: not %clang_cl --target=aarch64-unknown-uefi < %s 2>&1 | FileCheck -check-prefixes=CHECK-AARCH64 %s
// CHECK-AARCH64: error: unknown target triple 'aarch64-unknown-uefi'

// RUN: not %clang_cl --target=arm-unknown-uefi < %s 2>&1 | FileCheck -check-prefixes=CHECK-ARM %s
// CHECK-ARM: error: unknown target triple 'arm-unknown-uefi'

// RUN: not %clang_cl --target=x86-unknown-uefi < %s 2>&1 | FileCheck -check-prefixes=CHECK-x86 %s
// CHECK-x86: error: unknown target triple 'x86-unknown-uefi'
