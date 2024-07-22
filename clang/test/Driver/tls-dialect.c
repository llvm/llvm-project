// RUN: %clang -### --target=loongarch64-linux -mtls-dialect=trad %s 2>&1 | FileCheck --check-prefix=NODESC %s
// RUN: %clang -### --target=loongarch64-linux %s 2>&1 | FileCheck --check-prefix=NODESC %s
// RUN: %clang -### --target=riscv64-freebsd -mtls-dialect=desc %s 2>&1 | FileCheck --check-prefix=DESC %s
// RUN: %clang -### --target=riscv64-linux -mtls-dialect=trad %s 2>&1 | FileCheck --check-prefix=NODESC %s
// RUN: %clang -### --target=riscv64-linux %s 2>&1 | FileCheck --check-prefix=NODESC %s
// RUN: %clang -### --target=x86_64-linux -mtls-dialect=gnu %s 2>&1 | FileCheck --check-prefix=NODESC %s
// RUN: %clang -### --target=x86_64-linux -mtls-dialect=gnu2 %s 2>&1 | FileCheck --check-prefix=DESC %s

/// Android supports TLSDESC by default on RISC-V
/// TLSDESC is not on by default in Linux, even on RISC-V, and is covered above
// RUN: %clang -### --target=riscv64-android %s 2>&1 | FileCheck --check-prefix=DESC %s

/// LTO
// RUN: %clang -### --target=loongarch64-linux -flto -mtls-dialect=desc %s 2>&1 | FileCheck --check-prefix=LTO-DESC %s
// RUN: %clang -### --target=loongarch64-linux -flto %s 2>&1 | FileCheck --check-prefix=LTO-NODESC %s
// RUN: %clang -### --target=riscv64-linux -flto -mtls-dialect=desc %s 2>&1 | FileCheck --check-prefix=LTO-DESC %s
// RUN: %clang -### --target=riscv64-linux -flto %s 2>&1 | FileCheck --check-prefix=LTO-NODESC %s

/// Unsupported target
/// GCC supports -mtls-dialect= for AArch64, but we just unsupport it for AArch64 as it is very rarely used.
// RUN: not %clang --target=aarch64-linux -mtls-dialect=desc %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED-TARGET %s
// RUN: not %clang --target=x86_64-apple-macos -mtls-dialect=desc -flto %s 2>&1 | FileCheck -check-prefix=UNSUPPORTED-TARGET %s

/// Unsupported argument
// RUN: not %clang -### --target=loongarch64-linux -mtls-dialect=gnu2 %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED-ARG %s
// RUN: not %clang -### --target=riscv64-linux -mtls-dialect=gnu2 %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED-ARG %s

// DESC:       "-cc1" {{.*}}"-enable-tlsdesc"
// NODESC-NOT: "-enable-tlsdesc"
// LTO-DESC:       "-plugin-opt=-enable-tlsdesc"
// LTO-NODESC-NOT: "-plugin-opt=-enable-tlsdesc"

// UNSUPPORTED-TARGET: error: unsupported option '-mtls-dialect=' for target
// UNSUPPORTED-ARG: error: unsupported argument 'gnu2' to option '-mtls-dialect=' for target
