// RUN: %clang -fno-stack-protector -### %s 2>&1 | FileCheck %s -check-prefix=NOSSP
// NOSSP-NOT: "-stack-protector"
// NOSSP-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -target i386-unknown-linux -fstack-protector -### %s 2>&1 | FileCheck %s -check-prefix=SSP
// SSP: "-stack-protector" "1"
// SSP-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -target i386-unknown-linux -fstack-protector --param ssp-buffer-size=16 -### %s 2>&1 | FileCheck %s -check-prefix=SSP-BUF
// SSP-BUF: "-stack-protector" "1"
// SSP-BUF: "-stack-protector-buffer-size" "16" 

// RUN: %clang -target i386-pc-openbsd -### %s 2>&1 | FileCheck %s -check-prefix=OPENBSD
// OPENBSD: "-stack-protector" "2"

// RUN: %clang -target i386-pc-openbsd -fstack-protector -### %s 2>&1 | FileCheck %s -check-prefix=OPENBSD_SPS
// OPENBSD_SPS: "-stack-protector" "2"

// RUN: %clang -fstack-protector-strong -### %s 2>&1 | FileCheck %s -check-prefix=SSP-STRONG
// SSP-STRONG: "-stack-protector" "2"
// SSP-STRONG-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -fstack-protector-all -### %s 2>&1 | FileCheck %s -check-prefix=SSP-ALL
// SSP-ALL: "-stack-protector" "3"
// SSP-ALL-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -target x86_64-scei-ps4 -### %s 2>&1 | FileCheck %s -check-prefix=SSP-PS4
// RUN: %clang -target x86_64-scei-ps4 -fstack-protector -### %s 2>&1 | FileCheck %s -check-prefix=SSP-PS4
// SSP-PS4: "-stack-protector" "2"
// SSP-PS4-NOT: "-stack-protector-buffer-size"

// RUN: %clang -target x86_64-scei-ps4 -fstack-protector --param ssp-buffer-size=16 -### %s 2>&1 | FileCheck %s -check-prefix=SSP-PS4-BUF
// SSP-PS4-BUF: "-stack-protector" "2"
// SSP-PS4-BUF: "-stack-protector-buffer-size" "16"

// Test default stack protector values for Darwin platforms

// RUN: %clang -target armv7k-apple-watchos2.0 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_WATCHOS
// SSP_WATCHOS: "-stack-protector" "1"
// RUN: %clang -target arm64-apple-ios8.0.0 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_IOS
// SSP_IOS: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX
// SSP_MACOSX: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.5 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_5
// SSP_MACOSX_10_5: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.5 -mkernel -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_KERNEL
// SSP_MACOSX_KERNEL-NOT: "-stack-protector"
// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_6_KERNEL
// SSP_MACOSX_10_6_KERNEL: "-stack-protector" "1"

// Test default stack protector values for Darwin platforms with -ffreestanding

// RUN: %clang -ffreestanding -target armv7k-apple-watchos2.0 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_FREE_WATCHOS
// SSP_FREE_WATCHOS-NOT: "-stack-protector"
// RUN: %clang -ffreestanding -target arm64-apple-ios8.0.0 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_FREE_IOS
// SSP_FREE_IOS-NOT: "-stack-protector"
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacosx-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_FREE_MACOSX
// SSP_FREE_MACOSX-NOT: "-stack-protector"
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacosx-version-min=10.5 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_FREE_MACOSX_10_5
// SSP_FREE_MACOSX_10_5-NOT: "-stack-protector"
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacosx-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_FREE_MACOSX_10_6_KERNEL
// SSP_FREE_MACOSX_10_6_KERNEL-NOT: "-stack-protector"
