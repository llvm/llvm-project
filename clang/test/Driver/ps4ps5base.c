// REQUIRES: x86-registered-target

// Test that the driver always emits -fno-use-init-array on the PS4/PS5 targets
// since their ABI does not support the .init_array section.

// RUN: %clang -c %s -target x86_64-scei-ps4 -### 2>&1                      \
// RUN:   | FileCheck %s
// RUN: %clang -c %s -target x86_64-sie-ps5 -### 2>&1                       \
// RUN:   | FileCheck %s
// RUN: %clang -c %s -target x86_64-scei-ps4 -fno-use-init-array -### 2>&1  \
// RUN:   | FileCheck %s
// RUN: %clang -c %s -target x86_64-sie-ps5 -fno-use-init-array -### 2>&1   \
// RUN:   | FileCheck %s
// RUN: %clang -c %s -target x86_64-scei-ps4 -fuse-init-array -### 2>&1     \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: %clang -c %s -target x86_64-sie-ps5 -fuse-init-array -### 2>&1      \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR

// CHECK: "-fno-use-init-array"
// CHECK-NOT: "-fuse-init-array"

// CHECK-ERROR: unsupported option '-fuse-init-array' for target 'x86_64-{{(scei|sie)}}-ps{{[45]}}'
