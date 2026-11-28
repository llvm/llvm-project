// RUN: %clang -target arm64-apple-ios12.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-IOS-PRE13
// CHECK-IOS-PRE13-NOT: "-fstack-check"

// RUN: %clang -target arm64-apple-ios13.0 %s -### 2>&1 | FileCheck %s --check-prefix=IOS13
// IOS13: "-fstack-check"

// RUN: %clang -target arm64-apple-ios14.0-macabi %s -### 2>&1 | FileCheck %s --check-prefix=IOS14-MACABI
// IOS14-MACABI: "-fstack-check"

// RUN: %clang -target arm64-apple-macosx11.0 %s -### 2>&1 | FileCheck %s --check-prefix=OSX11
// OSX11: "-fstack-check"

// Stack checking is only enabled by default on arm64(e).
// RUN: %clang -target x86_64-apple-macosx15.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-X86
// CHECK-X86-NOT: "-fstack-check"

// RUN: %clang -target arm64-apple-watchos5.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-WATCH-PRE6
// CHECK-WATCH-PRE6-NOT: "-fstack-check"

// RUN: %clang -target arm64_32-apple-watchos6.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-WATCH-ARM64-32
// CHECK-WATCH-ARM64-32-NOT: "-fstack-check"

// RUN: %clang -target arm64-apple-watchos6.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-WATCH-6
// CHECK-WATCH-6: "-fstack-check"

// RUN: %clang -target arm64-apple-darwin %s -fstack-check -### 2>&1 | FileCheck %s --check-prefix=ENABLE
// ENABLE: "-fstack-check"

// RUN: %clang -target arm64-apple-darwin %s -fno-stack-check -### 2>&1 | FileCheck %s --check-prefix=DISABLE
// DISABLE-NOT: "-fno-stack-check"

// RUN: %clang -target arm64-apple-ios12.0 %s -nodefaultlibs -### 2>&1 | FileCheck %s --check-prefix=NODEFAULTLIBS
// NODEFAULTLIBS-NOT: "-fstack-check"

// RUN: %clang -target arm64-apple-ios12.0 %s -ffreestanding -### 2>&1 | FileCheck %s --check-prefix=FREESTANDING
// FREESTANDING-NOT: "-fstack-check"

// RUN: %clang -target arm64-apple-ios12.0 %s -fstack-check -ffreestanding -mkernel -### 2>&1 | FileCheck %s --check-prefix=FREESTANDING-KERNEL
// FREESTANDING-KERNEL: "-fstack-check"

// RUN: %clang -target arm64-apple-ios12.0 %s -fstack-check -ffreestanding -fapple-kext -### 2>&1 | FileCheck %s --check-prefix=FREESTANDING-KEXT
// FREESTANDING-KEXT: "-fstack-check"
