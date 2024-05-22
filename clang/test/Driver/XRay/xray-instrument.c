// RUN: %clang -### --target=aarch64-pc-freebsd -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -### --target=arm64-apple-macos -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64-apple-darwin -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clang -### --target=x86_64-pc-windows -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK: "-cc1" {{.*}}"-fxray-instrument"
// ERR:   error: unsupported option '-fxray-instrument' for target

typedef int a;
