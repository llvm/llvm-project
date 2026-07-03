// RUN: %clang -### --target=aarch64-pc-freebsd -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -### --target=arm64-apple-macos -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64-apple-darwin -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clang -### --target=x86_64-pc-windows -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

/// Checking -fxray-instrument with offloading and -Xarch_host
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -Xarch_host -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -x hip --offload-arch=gfx906 -nogpulib -nogpuinc -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -x hip --offload-arch=gfx906 -nogpulib -nogpuinc -Xarch_host -fxray-instrument -c %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: "-cc1" {{.*}}"-fxray-instrument"
// ERR:   error: unsupported option '-fxray-instrument' for target

typedef int a;
