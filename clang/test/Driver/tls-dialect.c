// RUN: %clang -### --target=riscv64-freebsd -mtls-dialect=desc %s 2>&1 | FileCheck --check-prefix=DESC %s
// RUN: %clang -### --target=riscv64-linux -mtls-dialect=trad %s 2>&1 | FileCheck --check-prefix=NODESC %s
// RUN: %clang -### --target=riscv64-linux %s 2>&1 | FileCheck --check-prefix=NODESC %s
// RUN: %clang -### --target=x86_64-linux -mtls-dialect=gnu %s 2>&1 | FileCheck --check-prefix=NODESC %s

/// Unsupported target
/// GCC supports -mtls-dialect= for AArch64, but we just unsupport it for AArch64 as it is very rarely used.
// RUN: not %clang --target=aarch64-linux -mtls-dialect=desc %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED-TARGET %s
// RUN: not %clang --target=x86_64-apple-macos -mtls-dialect=desc %s 2>&1 | FileCheck -check-prefix=UNSUPPORTED-TARGET %s

/// Unsupported argument
// RUN: not %clang -### --target=riscv64-linux -mtls-dialect=gnu2 %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED-ARG %s
// RUN: not %clang -### --target=x86_64-linux -mtls-dialect=gnu2 %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED-ARG %s

// DESC: "-cc1" {{.*}}"-enable-tlsdesc"
// NODESC-NOT: "-enable-tlsdesc"

// UNSUPPORTED-TARGET: error: unsupported option '-mtls-dialect=' for target
// UNSUPPORTED-ARG: error: unsupported argument 'gnu2' to option '-mtls-dialect=' for target
