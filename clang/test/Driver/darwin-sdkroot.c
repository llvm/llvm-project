// Check that SDKROOT is used to define the default for -isysroot on Darwin.
// REQUIRES: system-darwin
//
// RUN: rm -rf %t.tmpdir
// RUN: mkdir -p %t.tmpdir
// RUN: env SDKROOT=%t.tmpdir %clang -target x86_64-apple-darwin10 \
// RUN:   -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-BASIC < %t.log %s
//
// CHECK-BASIC: clang
// CHECK-BASIC: "-cc1"
// CHECK-BASIC: "-isysroot" "{{.*tmpdir}}"

// Check that we don't use SDKROOT as the default if it is not a valid path.
//
// RUN: rm -rf %t.nonpath
// RUN: env SDKROOT=%t.nonpath %clang -target x86_64-apple-darwin10 \
// RUN:   -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-NONPATH < %t.log %s
//
// CHECK-NONPATH: clang
// CHECK-NONPATH: "-cc1"
// CHECK-NONPATH-NOT: "-isysroot"

// Check that we don't use SDKROOT as the default if it is just "/"
//
// RUN: env SDKROOT=/ %clang -target x86_64-apple-darwin10 \
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
// RUN: env SDKROOT=%t/SDKs/iPhoneOS8.0.0.sdk %clang -target arm64-apple-darwin %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IPHONE %s
//
// CHECK-IPHONE: clang
// CHECK-IPHONE: "-cc1"
// CHECK-IPHONE: "-triple" "arm64-apple-ios8.0.0"
// CHECK-IPHONE: ld
// CHECK-IPHONE: "-iphoneos_version_min" "8.0.0"
// RUN: env SDKROOT=%t/SDKs/iPhoneOS8.0.0.sdk %clang %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IPHONE-X86 %s
// CHECK-IPHONE-X86: clang
// CHECK-IPHONE-X86: "-cc1"
// CHECK-IPHONE-X86: -apple-ios8.0.0"
// CHECK-IPHONE-X86: ld
// CHECK-IPHONE-X86: "-iphoneos_version_min" "8.0.0"
//
//
// RUN: rm -rf %t/SDKs/iPhoneSimulator8.0.sdk
// RUN: mkdir -p %t/SDKs/iPhoneSimulator8.0.sdk
// RUN: env SDKROOT=%t/SDKs/iPhoneSimulator8.0.sdk %clang -target x86_64-apple-darwin %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SIMULATOR %s
// RUN: env SDKROOT=%t/SDKs/iPhoneSimulator8.0.sdk %clang -arch x86_64 %s -### 2>&1 \
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
// RUN: env SDKROOT=%t/SDKs/MacOSX10.10.0.sdk %clang -target x86_64-apple-darwin %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MACOSX %s
//
// CHECK-MACOSX: clang
// CHECK-MACOSX: "-cc1"
// CHECK-MACOSX: "-triple" "x86_64-apple-macosx10.10.0"
// CHECK-MACOSX: ld
// CHECK-MACOSX: "-macosx_version_min" "10.10.0"

// RUN: rm -rf %t/SDKs/WatchOS3.0.sdk
// RUN: mkdir -p %t/SDKs/WatchOS3.0.sdk
// RUN: env SDKROOT=%t/SDKs/WatchOS3.0.sdk %clang %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WATCH %s
//
// CHECK-WATCH: clang
// CHECK-WATCH: "-cc1"
// CHECK-WATCH: -apple-watchos3.0.0"
// CHECK-WATCH: ld
// CHECK-WATCH: "-watchos_version_min" "3.0.0"
//
//
// RUN: rm -rf %t/SDKs/WatchSimulator3.0.sdk
// RUN: mkdir -p %t/SDKs/WatchSimulator3.0.sdk
// RUN: env SDKROOT=%t/SDKs/WatchSimulator3.0.sdk %clang %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WATCH-SIMULATOR %s
//
// CHECK-WATCH-SIMULATOR: clang
// CHECK-WATCH-SIMULATOR: "-cc1"
// CHECK-WATCH-SIMULATOR: -apple-watchos3.0.0-simulator"
// CHECK-WATCH-SIMULATOR: ld
// CHECK-WATCH-SIMULATOR: "-watchos_simulator_version_min" "3.0.0"

// RUN: rm -rf %t/SDKs/AppleTVOS10.0.sdk
// RUN: mkdir -p %t/SDKs/AppleTVOS10.0.sdk
// RUN: env SDKROOT=%t/SDKs/AppleTVOS10.0.sdk %clang %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-TV %s
//
// CHECK-TV: clang
// CHECK-TV: "-cc1"
// CHECK-TV: -apple-tvos10.0.0"
// CHECK-TV: ld
// CHECK-TV: "-tvos_version_min" "10.0.0"
//
//
// RUN: rm -rf %t/SDKs/AppleTVSimulator10.0.sdk
// RUN: mkdir -p %t/SDKs/AppleTVSimulator10.0.sdk
// RUN: env SDKROOT=%t/SDKs/AppleTVSimulator10.0.sdk %clang %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-TV-SIMULATOR %s
//
// CHECK-TV-SIMULATOR: clang
// CHECK-TV-SIMULATOR: "-cc1"
// CHECK-TV-SIMULATOR: -apple-tvos10.0.0-simulator"
// CHECK-TV-SIMULATOR: ld
// CHECK-TV-SIMULATOR: "-tvos_simulator_version_min" "10.0.0"
