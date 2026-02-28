// Check default enablement of Objective-C objc_msgSend selector stubs codegen.

// Enabled by default with ld64-811.2+ ...

// ... for arm64
// RUN: %clang -target arm64-apple-ios15            -mlinker-version=811.2 -### %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-ios15            -mlinker-version=811   -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS

// RUN: %clang -target arm64-apple-macos12          -mlinker-version=811.2 -### %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-macos12          -mlinker-version=811   -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS

// ... for arm64e
// RUN: %clang -target arm64e-apple-ios15           -mlinker-version=811.2 -### %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64e-apple-ios15           -mlinker-version=811   -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS

// ... and arm64_32.
// RUN: %clang -target arm64_32-apple-watchos8      -mlinker-version=811.2 -### %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64_32-apple-watchos8      -mlinker-version=811   -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS


// Disabled elsewhere, e.g. x86_64.
// RUN: %clang -target x86_64-apple-macos12         -mlinker-version=811.2 -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS
// RUN: %clang -target x86_64-apple-macos12         -mlinker-version=811   -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS

// RUN: %clang -target x86_64-apple-ios15-simulator -mlinker-version=811.2 -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS
// RUN: %clang -target x86_64-apple-ios15-simulator -mlinker-version=811   -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS

// ... or armv7k.
// RUN: %clang -target armv7k-apple-watchos6        -mlinker-version=811.2 -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS
// RUN: %clang -target armv7k-apple-watchos6        -mlinker-version=811   -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS


// Enabled if you ask for it.
// RUN: %clang -target arm64-apple-macos12 -fobjc-msgsend-selector-stubs                    -### %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-macos12 -fobjc-msgsend-selector-stubs -mlinker-version=0 -### %s 2>&1 | FileCheck %s

// Disabled if you ask for that.
// RUN: %clang -target arm64-apple-macos12 -fno-objc-msgsend-selector-stubs -mlinker-version=811.2 -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS


// CHECK: "-fobjc-msgsend-selector-stubs"
// NOSTUBS-NOT: objc-msgsend-selector-stubs
