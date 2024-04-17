// RUN: %clang -### -c --target=aarch64 -fno-ptrauth-intrinsics -fptrauth-intrinsics %s 2>&1 | FileCheck %s --check-prefix=INTRIN
// INTRIN: "-cc1"{{.*}} "-fptrauth-intrinsics"

// RUN: not %clang -### -c --target=x86_64 -fptrauth-intrinsics %s 2>&1 | FileCheck %s --check-prefix=ERR
// ERR: error: unsupported option '-fptrauth-intrinsics' for target '{{.*}}'
