// Check supported targets
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fPIC -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -### --target=aarch64-unknown-linux-gnu -fPIC -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s

// Check unsupported targets
// RUN: not %clang -### --target=arm-unknown-linux-gnu -fPIC -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-TARGET
// RUN: not %clang -### --target=mips-unknown-linux-gnu -fPIC -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-TARGET
// RUN: not %clang -### --target=loongarch64-unknown-linux-gnu -fPIC -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-TARGET
// RUN: not %clang -### --target=hexagon-unknown-linux-gnu -fPIC -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-TARGET
// RUN: not %clang -### --target=powerpc64le-unknown-linux-gnu -fPIC -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-TARGET

// Check PIC requirement
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fpic -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fno-PIC -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-PIC
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fno-pic -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-PIC
// On 64 bit darwin, PIC is always enabled
// RUN: %clang -### --target=x86_64-apple-darwin -fxray-instrument -fxray-shared -c %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: "-cc1" {{.*}}"-fxray-instrument" {{.*}}"-fxray-shared"
// ERR-TARGET:   error: unsupported option '-fxray-shared' for target
// ERR-PIC:   error: option '-fxray-shared' cannot be specified without '-fPIC'

