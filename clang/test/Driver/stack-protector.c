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
// RUN: %clang -target x86_64-apple-darwin10 -mmacos-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacos-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX
// SSP_MACOSX: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-darwin10 -mmacos-version-min=10.5 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_5
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacos-version-min=10.5 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_5
// SSP_MACOSX_10_5: "-stack-protector" "1"
// RUN: %clang -target x86_64-apple-darwin10 -mmacos-version-min=10.5 -mkernel -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_KERNEL
// SSP_MACOSX_KERNEL-NOT: "-stack-protector"
// RUN: %clang -target x86_64-apple-darwin10 -mmacos-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_6_KERNEL
// RUN: %clang -ffreestanding -target x86_64-apple-darwin10 -mmacos-version-min=10.6 -### %s 2>&1 | FileCheck %s -check-prefix=SSP_MACOSX_10_6_KERNEL
// SSP_MACOSX_10_6_KERNEL: "-stack-protector" "1"

// Test -fstack-protector-layout-only with the different stack protector levels

// RUN: %clang --target=x86_64-unknown-linux-gnu -fstack-protector-layout-only -### %s 2>&1 | FileCheck %s -check-prefix=SSP-LAYOUT-ONLY
// SSP-LAYOUT-ONLY: warning: overriding '-fno-stack-protector' option with '-fstack-protector-layout-only'
// SSP-LAYOUT-ONLY: "-stack-protector" "1" "-fstack-protector-layout-only"

// RUN: %clang --target=x86_64-unknown-linux-gnu -fno-stack-protector -fstack-protector-layout-only -### %s 2>&1 | FileCheck %s -check-prefix=SSP-LAYOUT-ONLY-NO-SSP
// RUN: %clang --target=x86_64-unknown-linux-gnu -fstack-protector-layout-only -fno-stack-protector -### %s 2>&1 | FileCheck %s -check-prefix=SSP-LAYOUT-ONLY-NO-SSP
// SSP-LAYOUT-ONLY-NO-SSP: warning: overriding '-fno-stack-protector' option with '-fstack-protector-layout-only'
// SSP-LAYOUT-ONLY-NO-SSP: "-stack-protector" "1" "-fstack-protector-layout-only"

// RUN: %clang --target=x86_64-unknown-linux-gnu -fstack-protector-layout-only -fstack-protector-strong -### %s 2>&1 | FileCheck %s -check-prefix=SSP-LAYOUT-ONLY-STRONG
// SSP-LAYOUT-ONLY-STRONG-NOT: warning:
// SSP-LAYOUT-ONLY-STRONG: "-stack-protector" "2" "-fstack-protector-layout-only"

// RUN: %clang --target=x86_64-unknown-linux-gnu -fstack-protector-all -fstack-protector-layout-only -### %s 2>&1 | FileCheck %s -check-prefix=SSP-LAYOUT-ONLY-ALL
// SSP-LAYOUT-ONLY-ALL-NOT: warning:
// SSP-LAYOUT-ONLY-ALL: "-stack-protector" "3" "-fstack-protector-layout-only"

// RUN: %clang --target=i386-pc-openbsd -fstack-protector-layout-only -### %s 2>&1 | FileCheck %s -check-prefix=SSP-LAYOUT-ONLY-OPENBSD
// SSP-LAYOUT-ONLY-OPENBSD-NOT: warning:
// SSP-LAYOUT-ONLY-OPENBSD: "-stack-protector" "2" "-fstack-protector-layout-only"

// RUN: %clang --target=x86_64-unknown-linux-gnu -fstack-protector-layout-only -fno-stack-protector-layout-only -### %s 2>&1 | FileCheck %s -check-prefix=NO-SSP-LAYOUT-ONLY
// NO-SSP-LAYOUT-ONLY-NOT: "-stack-protector"
// NO-SSP-LAYOUT-ONLY-NOT: "-fstack-protector-layout-only"

// RUN: %clang --target=bpf -fstack-protector-layout-only -### %s 2>&1 | FileCheck %s -check-prefix=SSP-LAYOUT-ONLY-BPF
// SSP-LAYOUT-ONLY-BPF: warning: ignoring '-fstack-protector-layout-only' option as it is not currently supported for target 'bpf'
// SSP-LAYOUT-ONLY-BPF-NOT: overriding
// SSP-LAYOUT-ONLY-BPF-NOT: "-stack-protector"
// SSP-LAYOUT-ONLY-BPF-NOT: "-fstack-protector-layout-only"
