// Check default enablement of Objective-C objc_msgSend selector stubs codegen.

// Enabled by default for AArch64 targets.

// arm64
// RUN: %clang -target arm64-apple-ios15            -### %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-macos12          -### %s 2>&1 | FileCheck %s

// arm64e
// RUN: %clang -target arm64e-apple-ios15           -### %s 2>&1 | FileCheck %s

// arm64_32
// RUN: %clang -target arm64_32-apple-watchos8      -### %s 2>&1 | FileCheck %s


// Disabled elsewhere, e.g. x86_64 ...
// RUN: %clang -target x86_64-apple-macos12         -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS
// RUN: %clang -target x86_64-apple-ios15-simulator -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS

// ... or armv7k.
// RUN: %clang -target armv7k-apple-watchos6        -### %s 2>&1 | FileCheck %s --check-prefix=NOSTUBS


// Disabled if you ask for that.
// RUN: %clang -target arm64-apple-macos12 -fno-objc-msgsend-selector-stubs       -### %s 2>&1 | FileCheck %s --check-prefix=CLASS_STUB_ONLY
// RUN: %clang -target arm64-apple-macos12 -fno-objc-msgsend-class-selector-stubs -### %s 2>&1 | FileCheck %s --check-prefix=INST_STUB_ONLY


// CHECK: "-fobjc-msgsend-selector-stubs" "-fobjc-msgsend-class-selector-stubs"
// INST_STUB_ONLY-NOT: objc-msgsend-class-selector-stubs
// INST_STUB_ONLY: objc-msgsend-selector-stubs
// INST_STUB_ONLY-NOT: objc-msgsend-class-selector-stubs
// CLASS_STUB_ONLY-NOT: objc-msgsend-selector-stubs
// CLASS_STUB_ONLY: objc-msgsend-class-selector-stubs
// CLASS_STUB_ONLY-NOT: objc-msgsend-selector-stubs
// NOSTUBS-NOT: objc-msgsend-{{.*}}selector-stubs
