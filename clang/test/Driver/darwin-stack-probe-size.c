// Darwin defaults the stack probe threshold to 16K on 16K-page (arm64) targets, and leaves it at the 4K default everywhere else.

// (1) arm64(e) Darwin + kernel/kext targets get 16k threshold
// RUN: %clang -target arm64-apple-macosx15.0 %s -### 2>&1 | FileCheck %s --check-prefix=PROBE16K
// RUN: %clang -target arm64e-apple-macosx15.0 %s -### 2>&1 | FileCheck %s --check-prefix=PROBE16K
// RUN: %clang -target arm64-apple-ios17.0 %s -### 2>&1 | FileCheck %s --check-prefix=PROBE16K
// RUN: %clang -target arm64-apple-tvos17.0 %s -### 2>&1 | FileCheck %s --check-prefix=PROBE16K
// RUN: %clang -target arm64-apple-watchos9.0 %s -### 2>&1 | FileCheck %s --check-prefix=PROBE16K
// RUN: %clang -target arm64-apple-macosx15.0 %s -mkernel -### 2>&1 | FileCheck %s --check-prefix=PROBE16K
// RUN: %clang -target arm64-apple-macosx15.0 %s -fapple-kext -### 2>&1 | FileCheck %s --check-prefix=PROBE16K
// PROBE16K: "-mstack-probe-size=16384"

// (2) x86_64 / arm64_32 / simulator / non-darwin keeps the 4k default
// RUN: %clang -target x86_64-apple-macosx15.0 %s -### 2>&1 | FileCheck %s --check-prefix=NOPROBEARG
// RUN: %clang -target x86_64-apple-ios17.0-simulator %s -fstack-check -### 2>&1 | FileCheck %s --check-prefix=NOPROBEARG
// RUN: %clang -target arm64_32-apple-watchos9.0 %s -### 2>&1 | FileCheck %s --check-prefix=NOPROBEARG
// RUN: %clang -target arm64-apple-ios17.0-simulator %s -### 2>&1 | FileCheck %s --check-prefix=NOPROBEARG
// RUN: %clang -target aarch64-linux-gnu %s -### 2>&1 | FileCheck %s --check-prefix=NOPROBEARG
// RUN: %clang -target aarch64-linux-gnu %s -fstack-clash-protection -### 2>&1 | FileCheck %s --check-prefix=NOPROBEARG
// NOPROBEARG-NOT: "-mstack-probe-size=

// (3) -ffreestanding keeps the 4k default (except for kernel/kext)
// RUN: %clang -target arm64-apple-macosx15.0 %s -ffreestanding -### 2>&1 | FileCheck %s --check-prefix=FREESTANDING-NOPROBEARG
// RUN: %clang -target arm64-apple-macosx15.0 %s -ffreestanding -fstack-check -### 2>&1 | FileCheck %s --check-prefix=FREESTANDING-NOPROBEARG
// RUN: %clang -target arm64-apple-ios17.0 %s -ffreestanding -fstack-check -### 2>&1 | FileCheck %s --check-prefix=FREESTANDING-NOPROBEARG
// FREESTANDING-NOPROBEARG-NOT: "-mstack-probe-size=

// (4) firmware triples keep the 4k default, with or without -ffreestanding.
// RUN: %clang -target arm64-apple-firmware %s -### 2>&1 | FileCheck %s --check-prefix=FIRMWARE-NOPROBEARG
// RUN: %clang -target arm64e-apple-firmware %s -### 2>&1 | FileCheck %s --check-prefix=FIRMWARE-NOPROBEARG
// RUN: %clang -target arm64-apple-firmware %s -fstack-check -### 2>&1 | FileCheck %s --check-prefix=FIRMWARE-NOPROBEARG
// RUN: %clang -target arm64-apple-firmware %s -ffreestanding -fstack-check -### 2>&1 | FileCheck %s --check-prefix=FIRMWARE-NOPROBEARG
// RUN: %clang -target arm64-apple-firmware %s -mkernel -### 2>&1 | FileCheck %s --check-prefix=FIRMWARE-NOPROBEARG
// FIRMWARE-NOPROBEARG-NOT: "-mstack-probe-size=

// (5) explicit -mstack-probe-size always wins
// RUN: %clang -target arm64-apple-macosx15.0 %s -mstack-probe-size=4096 -### 2>&1 | FileCheck %s --check-prefix=PROBE4K
// RUN: %clang -target arm64-apple-ios17.0 %s -mstack-probe-size=4096 -### 2>&1 | FileCheck %s --check-prefix=PROBE4K
// RUN: %clang -target arm64-apple-macosx15.0 %s -mstack-probe-size=8192 -mstack-probe-size=4096 -### 2>&1 | FileCheck %s --check-prefix=PROBE4K
// PROBE4K: "-mstack-probe-size=4096"
// PROBE4K-NOT: "-mstack-probe-size=16384"
// RUN: %clang -target arm64-apple-macosx15.0 %s -mstack-probe-size=8192 -### 2>&1 | FileCheck %s --check-prefix=PROBE8K
// PROBE8K: "-mstack-probe-size=8192"
// PROBE8K-NOT: "-mstack-probe-size=16384"
// RUN: %clang -target arm64-apple-macosx15.0 %s -mstack-probe-size=65536 -### 2>&1 | FileCheck %s --check-prefix=PROBE64K
// PROBE64K: "-mstack-probe-size=65536"
// RUN: %clang -target arm64-apple-macosx15.0 %s -mstack-probe-size=16384 -### 2>&1 | FileCheck %s --check-prefix=PROBE16K-ONCE
// PROBE16K-ONCE: "-mstack-probe-size=16384"
// PROBE16K-ONCE-NOT: "-mstack-probe-size=

// (6) explicit value opts in and always wins (kernel/kext, -ffreestanding, firmware)
// RUN: %clang -target arm64-apple-macosx15.0 %s -mkernel -mstack-probe-size=4096 -### 2>&1 | FileCheck %s --check-prefix=OPT-IN-PROBE4K
// RUN: %clang -target arm64-apple-macosx15.0 %s -fapple-kext -mstack-probe-size=4096 -### 2>&1 | FileCheck %s --check-prefix=OPT-IN-PROBE4K
// RUN: %clang -target arm64-apple-macosx15.0 %s -ffreestanding -mstack-probe-size=16384 -### 2>&1 | FileCheck %s --check-prefix=OPT-IN-PROBE16K
// RUN: %clang -target arm64-apple-firmware %s -mstack-probe-size=16384 -### 2>&1 | FileCheck %s --check-prefix=OPT-IN-PROBE16K
// OPT-IN-PROBE4K: "-mstack-probe-size=4096"
// OPT-IN-PROBE4K-NOT: "-mstack-probe-size=16384"
// OPT-IN-PROBE16K: "-mstack-probe-size=16384"

void f(void) {}
