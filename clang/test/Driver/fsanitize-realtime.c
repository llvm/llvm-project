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

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=realtime,thread  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-REALTIME-TSAN
// CHECK-REALTIME-TSAN: error: invalid argument '-fsanitize=realtime' not allowed with '-fsanitize=thread'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=realtime,address  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-REALTIME-ASAN
// CHECK-REALTIME-ASAN: error: invalid argument '-fsanitize=realtime' not allowed with '-fsanitize=address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=realtime,memory  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-REALTIME-MSAN
// CHECK-REALTIME-MSAN: error: invalid argument '-fsanitize=realtime' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=realtime,undefined  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-REALTIME-UBSAN
// CHECK-REALTIME-UBSAN: error: invalid argument '-fsanitize=realtime' not allowed with '-fsanitize=undefined'
