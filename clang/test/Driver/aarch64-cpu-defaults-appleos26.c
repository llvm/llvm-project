/// iOS 26 and watchOS 26 bump the default arm64 CPU targets.

/// arm64 iOS 26 defaults to apple-a12.  arm64e already did.
// RUN: %clang -target arm64-apple-ios26  -### -c %s 2>&1 | FileCheck %s --check-prefix=A12
// RUN: %clang -target arm64e-apple-ios26 -### -c %s 2>&1 | FileCheck %s --check-prefix=A12

/// arm64e/arm64_32 watchOS 26 default to apple-s6.
// RUN: %clang -target arm64e-apple-watchos26   -### -c %s 2>&1 | FileCheck %s --check-prefix=S6
// RUN: %clang -target arm64_32-apple-watchos26 -### -c %s 2>&1 | FileCheck %s --check-prefix=S6

/// arm64 is new in watchOS 26, and defaults to apple-s6.
// RUN: %clang -target arm64-apple-watchos26  -### -c %s 2>&1 | FileCheck %s --check-prefix=S6

/// llvm usually treats tvOS like iOS, but it runs on different hardware.
// RUN: %clang -target arm64-apple-tvos26  -### -c %s 2>&1 | FileCheck %s --check-prefix=A7
// RUN: %clang -target arm64e-apple-tvos26 -### -c %s 2>&1 | FileCheck %s --check-prefix=A12

/// Simulators are tested with other Mac-like targets in aarch64-mac-cpus.c.

// A12: "-target-cpu" "apple-a12"
// S6:  "-target-cpu" "apple-s6"
// A7:  "-target-cpu" "apple-a7"
