// RUN: %clang -target x86_64-apple-macosx10.13 -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=UNAVAILABLE

// RUN: %clang -target arm64-apple-ios10 -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=UNAVAILABLE

// RUN: %clang -target arm64-apple-tvos10 -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=UNAVAILABLE

// RUN: %clang -target thumbv7-apple-watchos3 -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=UNAVAILABLE

// RUN: %clang -target x86_64-apple-darwin -mios-simulator-version-min=10 \
// RUN:  -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=UNAVAILABLE

// RUN: %clang -target x86_64-apple-darwin -mtvos-simulator-version-min=10 \
// RUN: -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=UNAVAILABLE

// RUN: %clang -target x86_64-apple-darwin -mwatchos-simulator-version-min=3 \
// RUN: -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=UNAVAILABLE

// RUN: %clang -target s390x-none-zos -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=UNAVAILABLE

// UNAVAILABLE: "-faligned-alloc-unavailable"

// RUN: %clang -target x86_64-apple-macosx10.14 -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target arm64-apple-ios11 -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target arm64-apple-tvos11 -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target armv7k-apple-watchos4 -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target x86_64-unknown-linux-gnu -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target x86_64-apple-darwin -mios-simulator-version-min=11 \
// RUN:  -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target x86_64-apple-darwin -mtvos-simulator-version-min=11 \
// RUN: -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target x86_64-apple-darwin -mwatchos-simulator-version-min=4 \
// RUN: -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// Check that passing -faligned-allocation or -fno-aligned-allocation stops the
// driver from passing -faligned-alloc-unavailable to cc1.

// RUN: %clang -target x86_64-apple-macosx10.13 -faligned-allocation -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target x86_64-apple-macosx10.13 -fno-aligned-allocation -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target s390x-none-zos -faligned-allocation -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// RUN: %clang -target s390x-none-zos -fno-aligned-allocation -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=AVAILABLE

// AVAILABLE-NOT: "-faligned-alloc-unavailable"
