// Check that -mlfi= lowers LFI configuration to subtarget features.

// RUN: %clang --target=aarch64_lfi-linux -mlfi=no-loads,no-stores -c %s -### 2>&1 | FileCheck %s
// CHECK-DAG: "-target-feature" "+no-lfi-loads"
// CHECK-DAG: "-target-feature" "+no-lfi-stores"

// RUN: not %clang --target=aarch64_lfi-linux -mlfi=unknown -c %s -### 2>&1 | FileCheck %s --check-prefix=BAD
// BAD: unsupported argument 'unknown' to option '-mlfi='

// RUN: not %clang --target=aarch64-linux -mlfi=no-loads -c %s -### 2>&1 | FileCheck %s --check-prefix=NOLFI
// NOLFI: unsupported option '-mlfi=' for target
