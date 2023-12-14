// On Darwin, arch should override CPU for triple purposes
// RUN: %clang -target armv7m-apple-darwin -arch armv7m -mcpu=cortex-m4 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7M-DARWIN %s
// CHECK-V7M-DARWIN: "-cc1"{{.*}} "-triple" "thumbv7m-{{.*}} "-target-cpu" "cortex-m4"

/// -arch is unsupported for non-Darwin targets.
// RUN: not %clang --target=armv7m -arch armv7m -mcpu=cortex-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ERR %s
// ERR: unsupported option '-arch' for target 'armv7m'

// RUN: not %clang --target=aarch64-linux-gnu -arch arm64 -### -c %s 2>&1 | FileCheck -check-prefix=ERR2 %s
// ERR2: unsupported option '-arch' for target 'aarch64-linux-gnu'
