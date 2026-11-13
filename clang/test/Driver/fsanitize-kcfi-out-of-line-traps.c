// RUN: %clang --target=x86_64-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=kcfi -fno-sanitize-kcfi-out-of-line-traps -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s
// CHECK: "-fsanitize-kcfi-out-of-line-traps"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=INLINE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps -fno-sanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=INLINE
// RUN: %clang --target=riscv64-linux-gnu -fsanitize=kcfi -fno-sanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=INLINE
// INLINE-NOT: "-fsanitize-kcfi-out-of-line-traps"

// RUN: not %clang --target=riscv64-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=TARGET
// TARGET: error: unsupported option '-fsanitize-kcfi-out-of-line-traps' for target 'riscv64-unknown-linux-gnu'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=UNUSED
// UNUSED: warning: argument unused during compilation: '-fsanitize-kcfi-out-of-line-traps'
