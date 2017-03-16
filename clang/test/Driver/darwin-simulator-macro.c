// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -mios-simulator-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=SIM %s
// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -miphonesimulator-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=SIM %s
// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -mtvos-simulator-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=SIM %s
// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -mappletvsimulator-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=SIM %s
// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -mwatchos-simulator-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=SIM %s
// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -mwatchsimulator-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=SIM %s
// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -mios-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=DEV %s
// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -mtvos-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=DEV %s
// RUN: %clang -target x86_64-apple-darwin10 -arch x86_64 -mwatchos-version-min=6.0.0 -E -dM %s | FileCheck -check-prefix=DEV %s

// SIM: #define __APPLE_EMBEDDED_SIMULATOR__ 1
// DEV-NOT: __APPLE_EMBEDDED_SIMULATOR__
