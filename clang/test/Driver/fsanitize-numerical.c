// RUN: %clang --target=x86_64-linux-gnu -fsanitize=numerical %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NSAN-X86-64-LINUX
// CHECK-NSAN-X86-64-LINUX: "-fsanitize=numerical"

// RUN: not %clang --target=aarch64-unknown-linux-gnu -fsanitize=numerical %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NSAN-AARCH64-LINUX
// CHECK-NSAN-AARCH64-LINUX: error: unsupported option '-fsanitize=numerical' for target 'aarch64-unknown-linux-gnu'

// RUN: not %clang --target=mips-unknown-linux -fsanitize=numerical %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NSAN-MIPS-LINUX
// CHECK-NSAN-MIPS-LINUX: error: unsupported option '-fsanitize=numerical' for target 'mips-unknown-linux'

// RUN: %clang --target=x86_64-apple-macos -fsanitize=numerical %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NSAN-X86-64-MACOS
// CHECK-NSAN-X86-64-MACOS: "-fsanitize=numerical"

// RUN: not %clang --target=arm64-apple-macos -fsanitize=numerical %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NSAN-ARM64-MACOS
// CHECK-NSAN-ARM64-MACOS: error: unsupported option '-fsanitize=numerical' for target 'arm64-apple-macos'
