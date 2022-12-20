// This test will fail if CLANG_DEFAULT_CXX_STDLIB is set to libstdc++.
// XFAIL: default-cxx-stdlib=libstdc++

// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch arm64 -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -mmacosx-version-min=10.9 %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7s -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7k %s -### 2>&1 | FileCheck %s

// CHECK: "-stdlib=libc++"
// CHECK-NOT: "-stdlib=libstdc++"
