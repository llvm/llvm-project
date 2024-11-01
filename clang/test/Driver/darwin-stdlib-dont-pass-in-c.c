// This test ensures that the Driver doesn't pass -stdlib= down to CC1 when compiling C code.

// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch arm64 -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -mmacosx-version-min=10.9 %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7s -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7k %s -### 2>&1 | FileCheck %s

// CHECK-NOT: "-stdlib="
