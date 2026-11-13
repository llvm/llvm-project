// RUN: %clang -c --target=x86_64-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s
// RUN: %clang -c --target=aarch64-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s
// RUN: %clang -c --target=x86_64-linux-gnu -fsanitize=kcfi -fno-sanitize-kcfi-out-of-line-traps -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s
// CHECK: "-fsanitize-kcfi-out-of-line-traps"

// RUN: %clang -c --target=x86_64-linux-gnu -fsanitize=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=INLINE --implicit-check-not="-fsanitize-kcfi-out-of-line-traps"
// RUN: %clang -c --target=x86_64-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps -fno-sanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=INLINE --implicit-check-not="-fsanitize-kcfi-out-of-line-traps"
// RUN: %clang -c --target=riscv64-linux-gnu -fsanitize=kcfi -fno-sanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=INLINE --implicit-check-not="-fsanitize-kcfi-out-of-line-traps"
// INLINE: "-cc1"{{.*}}"-fsanitize=kcfi"

// RUN: not %clang -c --target=riscv64-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=TARGET-RISCV
// TARGET-RISCV: error: unsupported option '-fsanitize-kcfi-out-of-line-traps' for target 'riscv64-unknown-linux-gnu'
// RUN: not %clang -c --target=i386-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=TARGET-I386
// TARGET-I386: error: unsupported option '-fsanitize-kcfi-out-of-line-traps' for target 'i386-unknown-linux-gnu'

// RUN: %clang -c --target=x86_64-linux-gnu -fsanitize-kcfi-out-of-line-traps %s -### 2>&1 | FileCheck %s --check-prefix=UNUSED
// UNUSED: warning: argument unused during compilation: '-fsanitize-kcfi-out-of-line-traps'
// UNUSED-NOT: "-fsanitize-kcfi-out-of-line-traps"
