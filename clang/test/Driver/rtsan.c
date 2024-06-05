// RUN: %clang --target=x86_64-apple-darwin -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-X86-64-DARWIN
// CHECK-RTSAN-X86-64-DARWIN-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-darwin -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-X86-64-DARWIN
// CHECK-RTSAN-X86-64-DARWIN-NOT: unsupported option
// RUN: %clang --target=x86_64-apple-macos -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-X86-64-MACOS
// CHECK-RTSAN-X86-64-MACOS-NOT: unsupported option
// RUN: %clang --target=arm64-apple-macos -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-ARM64-MACOS
// CHECK-RTSAN-ARM64-MACOS-NOT: unsupported option

// RUN: %clang --target=arm64-apple-ios-simulator -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-ARM64-IOSSIMULATOR
// CHECK-RTSAN-ARM64-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=arm64-apple-watchos-simulator -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-ARM64-WATCHOSSIMULATOR
// CHECK-RTSAN-ARM64-WATCHOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=arm64-apple-tvos-simulator -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-ARM64-TVOSSIMULATOR
// CHECK-RTSAN-ARM64-TVOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-ios-simulator -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-X86-64-IOSSIMULATOR
// CHECK-RTSAN-X86-64-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-watchos-simulator -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-X86-64-WATCHOSSIMULATOR
// CHECK-RTSAN-X86-64-WATCHOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-tvos-simulator -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-X86-64-TVOSSIMULATOR
// CHECK-RTSAN-X86-64-TVOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-X86-64-LINUX
// CHECK-RTSAN-X86-64-LINUX-NOT: unsupported option

// RUN: not %clang --target=i386-pc-openbsd -fsanitize=realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN-OPENBSD
// CHECK-RTSAN-OPENBSD: unsupported option '-fsanitize=realtime' for target 'i386-pc-openbsd'
