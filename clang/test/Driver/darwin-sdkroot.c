// Check that SDKROOT is used to define the default for -isysroot on Darwin.
//
// RUN: rm -rf %t.tmpdir
// RUN: mkdir -p %t.tmpdir
// RUN: env SDKROOT=%t.tmpdir %clang -target x86_64-apple-darwin10 --sysroot="" \
// RUN:   -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-BASIC < %t.log %s
//
// CHECK-BASIC: clang
// CHECK-BASIC: "-cc1"
// CHECK-BASIC: "-isysroot" "{{.*tmpdir}}"

// Check that we don't use SDKROOT as the default if it is not a valid path.
//
// RUN: rm -rf %t.nonpath
// RUN: env SDKROOT=%t.nonpath %clang -target x86_64-apple-darwin10 --sysroot="" \
// RUN:   -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-NONPATH < %t.log %s
//
// CHECK-NONPATH: clang
// CHECK-NONPATH: "-cc1"
// CHECK-NONPATH-NOT: "-isysroot"

// Check that we don't use SDKROOT as the default if it is just "/"
//
// RUN: env SDKROOT=/ %clang -target x86_64-apple-darwin10 --sysroot="" \
// RUN:   -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-NONROOT < %t.log %s
//
// CHECK-NONROOT: clang
// CHECK-NONROOT: "-cc1"
// CHECK-NONROOT-NOT: "-isysroot"
//
// This test fails with MSYS or MSYS2 env.exe, since it does not preserve
// root, expanding / into C:/MINGW/MSYS/1.0 or c:/msys64. To reproduce the
// problem, run:
//
//   env SDKROOT=/ cmd //c echo %SDKROOT%
//
// This test do pass using GnuWin32 env.exe.

// Check if clang set the correct deployment target from -sysroot
//
// RUN: rm -rf %t/SDKs/iPhoneOS8.0.0.sdk
// RUN: mkdir -p %t/SDKs/iPhoneOS8.0.0.sdk
// RUN: env SDKROOT=%t/SDKs/iPhoneOS8.0.0.sdk %clang -fuse-ld= -target arm64-apple-darwin -mlinker-version=400 --sysroot="" %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IPHONE %s
//
// CHECK-IPHONE: clang
// CHECK-IPHONE: "-cc1"
// CHECK-IPHONE: "-triple" "arm64-apple-ios8.0.0"
// CHECK-IPHONE: ld
// CHECK-IPHONE: "-iphoneos_version_min" "8.0.0"
//
//
// RUN: rm -rf %t/SDKs/iPhoneSimulator8.0.sdk
// RUN: mkdir -p %t/SDKs/iPhoneSimulator8.0.sdk
// RUN: env SDKROOT=%t/SDKs/iPhoneSimulator8.0.sdk %clang -fuse-ld= -target x86_64-apple-darwin -mlinker-version=400 --sysroot="" %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SIMULATOR %s
//
// CHECK-SIMULATOR: clang
// CHECK-SIMULATOR: "-cc1"
// CHECK-SIMULATOR: "-triple" "x86_64-apple-ios8.0.0-simulator"
// CHECK-SIMULATOR: ld
// CHECK-SIMULATOR: "-ios_simulator_version_min" "8.0.0"
//
// RUN: rm -rf %t/SDKs/MacOSX10.10.0.sdk
// RUN: mkdir -p %t/SDKs/MacOSX10.10.0.sdk
// RUN: env SDKROOT=%t/SDKs/MacOSX10.10.0.sdk %clang -fuse-ld= -target x86_64-apple-darwin -mlinker-version=400 --sysroot="" %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MACOSX %s
//
// CHECK-MACOSX: clang
// CHECK-MACOSX: "-cc1"
// CHECK-MACOSX: "-triple" "x86_64-apple-macosx10.10.0"
// CHECK-MACOSX: ld
// CHECK-MACOSX: "-macosx_version_min" "10.10.0"
