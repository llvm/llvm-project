// REQUIRES: aarch64-registered-target

// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-intrinsics %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-calls %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-returns %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-indirect-gotos %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-auth-traps %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-vtable-pointer-address-discrimination %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-vtable-pointer-type-discrimination %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-function-pointer-type-discrimination %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-init-fini %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-init-fini-address-discrimination %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -faarch64-jump-table-hardening %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-objc-isa %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-objc-class-ro %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=aarch64 -fptrauth-objc-interface-sel %s -c  2>&1 | FileCheck %s
// RUN: not %clang  -pedantic-errors --target=arm64e %s -c  2>&1 | FileCheck %s --check-prefix ARM64E_TRIPLE
// RUN: not %clang  -pedantic-errors --target=arm64e-apple-macosx10.0 %s -c  2>&1 | FileCheck %s --check-prefix ARM64E_MACOS_TRIPLE
// RUN: not %clang  -pedantic-errors -arch arm64e %s -c  2>&1 | FileCheck %s --check-prefix ARM64E_ARCH

// FIXME: Cannot work out what the correct invocation to permit -fptrauth-elf-got is
// -- not %clang  -pedantic-errors --target=aarch64 -fptrauth-elf-got %s -c  2>&1 | FileCheck %s

int i;

// CHECK: error: the combination of '{{.*}}' and '-pedantic-errors' is incompatible
// ARM64E_TRIPLE: error: unsupported option '-pedantic-errors' for target 'arm64e'
// ARM64E_MACOS_TRIPLE: error: unsupported option '-pedantic-errors' for target 'arm64e-apple-macosx10.0.0'

// We have a trailing 'arm64e with no closing ' as the full triple is inferred from the host
// which we don't care about, and don't want to specify as we're wanting to ensure that *just*
// using '-arch arm64e' is sufficient
// ARM64E_ARCH: error: unsupported option '-pedantic-errors' for target 'arm64e