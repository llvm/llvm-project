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
// RUN: %clang -ffreestanding -target armv7k-apple-watchos2.0 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_WATCHOS
// SSP_WATCHOS: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-driverkit19.0 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_DRIVERKIT
// SSP_DRIVERKIT: "-stack-protector" "1"
// RUN: %clang -target arm64-apple-ios8.0.0 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_IOS
// RUN: %clang -ffreestanding -target arm64-apple-ios8.0.0 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_IOS
// SSP_IOS: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacosx-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX
// SSP_MACOSX: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.5 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_5
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacosx-version-min=10.5 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_5
// SSP_MACOSX_10_5: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.5 -mkernel -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_KERNEL
// SSP_MACOSX_KERNEL-NOT: "-stack-protector"
// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_6_KERNEL
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacosx-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_6_KERNEL
// SSP_MACOSX_10_6_KERNEL: "-stack-protector" "1"

