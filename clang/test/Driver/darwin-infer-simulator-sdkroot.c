// Check that SDKROOT does not infer simulator on when it points to a regular
// SDK.
// REQUIRES: system-darwin && native
//
// RUN: rm -rf %t/SDKs/iPhoneOS8.0.0.sdk
// RUN: mkdir -p %t/SDKs/iPhoneOS8.0.0.sdk
// RUN: env SDKROOT=%t/SDKs/iPhoneOS8.0.0.sdk %clang %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IPHONE %s
// RUN: env SDKROOT=%t/SDKs/iPhoneOS8.0.0.sdk IPHONEOS_DEPLOYMENT_TARGET=8.0 %clang %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IPHONE %s
// CHECK-IPHONE: clang
// CHECK-IPHONE: "-cc1"
// CHECK-IPHONE: -apple-ios8.0.0"
// CHECK-IPHONE: ld
// CHECK-IPHONE: "-iphoneos_version_min" "8.0.0"
//
//
// RUN: rm -rf %t/SDKs/iPhoneSimulator8.0.sdk
// RUN: mkdir -p %t/SDKs/iPhoneSimulator8.0.sdk
// RUN: env SDKROOT=%t/SDKs/iPhoneSimulator8.0.sdk %clang -arch x86_64 %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SIMULATOR %s
//
// CHECK-SIMULATOR: clang
// CHECK-SIMULATOR: "-cc1"
// CHECK-SIMULATOR: -apple-ios8.0.0-simulator"
// CHECK-SIMULATOR: ld
// CHECK-SIMULATOR: "-ios_simulator_version_min" "8.0.0"
//
//
// RUN: rm -rf %t/SDKs/iPhoneSimulator14.0.sdk
// RUN: mkdir -p %t/SDKs/iPhoneSimulator14.0.sdk
// RUN: env SDKROOT=%t/SDKs/iPhoneSimulator14.0.sdk %clang -arch arm64 %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SIMULATOR-ARM64 %s
//
// CHECK-SIMULATOR-ARM64: clang
// CHECK-SIMULATOR-ARM64: "-cc1"
// CHECK-SIMULATOR-ARM64: -apple-ios14.0.0-simulator"
// CHECK-SIMULATOR-ARM64: ld
// CHECK-SIMULATOR-ARM64: "-ios_simulator_version_min" "14.0.0"
//
//
// RUN: rm -rf %t/SDKs/WatchOS3.0.sdk
// RUN: mkdir -p %t/SDKs/WatchOS3.0.sdk
// RUN: env SDKROOT=%t/SDKs/WatchOS3.0.sdk %clang %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WATCH %s
// RUN: env WATCHOS_DEPLOYMENT_TARGET=3.0 %clang %s -isysroot %t/SDKs/WatchOS3.0.sdk -mlinker-version=400 -### 2>&1 \
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
// RUN: env SDKROOT=%t/SDKs/WatchSimulator3.0.sdk %clang -arch x86_64 %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WATCH-SIMULATOR %s
//
// CHECK-WATCH-SIMULATOR: clang
// CHECK-WATCH-SIMULATOR: "-cc1"
// CHECK-WATCH-SIMULATOR: -apple-watchos3.0.0-simulator"
// CHECK-WATCH-SIMULATOR: ld
// CHECK-WATCH-SIMULATOR: "-watchos_simulator_version_min" "3.0.0"
//
//
// RUN: rm -rf %t/SDKs/WatchSimulator7.0.sdk
// RUN: mkdir -p %t/SDKs/WatchSimulator7.0.sdk
// RUN: env SDKROOT=%t/SDKs/WatchSimulator7.0.sdk %clang -arch arm64 %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WATCH-SIMULATOR-ARM64 %s
//
// CHECK-WATCH-SIMULATOR-ARM64: clang
// CHECK-WATCH-SIMULATOR-ARM64: "-cc1"
// CHECK-WATCH-SIMULATOR-ARM64: -apple-watchos7.0.0-simulator"
// CHECK-WATCH-SIMULATOR-ARM64: ld
// CHECK-WATCH-SIMULATOR-ARM64: "-watchos_simulator_version_min" "7.0.0"
//
//
// RUN: rm -rf %t/SDKs/AppleTVOS10.0.sdk
// RUN: mkdir -p %t/SDKs/AppleTVOS10.0.sdk
// RUN: env SDKROOT=%t/SDKs/AppleTVOS10.0.sdk %clang %s -mlinker-version=400 -### 2>&1 \
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
// RUN: env SDKROOT=%t/SDKs/AppleTVSimulator10.0.sdk %clang -arch x86_64 %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-TV-SIMULATOR %s
//
// CHECK-TV-SIMULATOR: clang
// CHECK-TV-SIMULATOR: "-cc1"
// CHECK-TV-SIMULATOR: -apple-tvos10.0.0-simulator"
// CHECK-TV-SIMULATOR: ld
// CHECK-TV-SIMULATOR: "-tvos_simulator_version_min" "10.0.0"
//
//
// RUN: rm -rf %t/SDKs/AppleTVSimulator14.0.sdk
// RUN: mkdir -p %t/SDKs/AppleTVSimulator14.0.sdk
// RUN: env SDKROOT=%t/SDKs/AppleTVSimulator14.0.sdk %clang -arch arm64 %s -mlinker-version=400 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-TV-SIMULATOR-ARM64 %s
//
// CHECK-TV-SIMULATOR-ARM64: clang
// CHECK-TV-SIMULATOR-ARM64: "-cc1"
// CHECK-TV-SIMULATOR-ARM64: -apple-tvos14.0.0-simulator"
// CHECK-TV-SIMULATOR-ARM64: ld
// CHECK-TV-SIMULATOR-ARM64: "-tvos_simulator_version_min" "14.0.0"

