// RUN: not %clang -### -c --target=i686 -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=ERR-MEDIUM %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=tiny %s 2>&1 | FileCheck --check-prefix=TINY %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=small %s 2>&1 | FileCheck --check-prefix=SMALL %s
// RUN: %clang --target=x86_64 -### -S -mcmodel=kernel %s 2>&1 | FileCheck --check-prefix=KERNEL %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=MEDIUM %s
// RUN: %clang --target=x86_64 -### -S -mcmodel=large %s 2>&1 | FileCheck --check-prefix=LARGE %s
// RUN: not %clang -### -c --target=powerpc-linux-gnu -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=ERR-MEDIUM %s
// RUN: %clang --target=powerpc-unknown-aix -### -S -mcmodel=medium %s 2> %t.log
// RUN: FileCheck --check-prefix=AIX-MCMEDIUM-OVERRIDE %s < %t.log
// RUN: not %clang -### -c -mcmodel=lager %s 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: %clang --target=aarch64 -### -S -mcmodel=large -fno-pic %s 2>&1 | FileCheck --check-prefix=LARGE %s
// RUN: not %clang --target=aarch64 -### -S -mcmodel=large -fpic %s 2>&1 | FileCheck --check-prefix=AARCH64-PIC-LARGE %s
// RUN: not %clang -### -c --target=aarch64 -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=ERR-MEDIUM %s
// RUN: not %clang -### -c --target=aarch64 -mcmodel=kernel %s 2>&1 | FileCheck --check-prefix=ERR-KERNEL %s
// RUN: not %clang --target=aarch64_32-linux -### -S -mcmodel=small %s 2>&1 | FileCheck --check-prefix=ERR-AARCH64_32 %s

// TINY: "-mcmodel=tiny"
// SMALL: "-mcmodel=small"
// KERNEL: "-mcmodel=kernel"
// MEDIUM: "-mcmodel=medium"
// LARGE: "-mcmodel=large"
// AIX-MCMEDIUM-OVERRIDE: "-mcmodel=large"

// INVALID: error: unsupported argument 'lager' to option '-mcmodel=' for target '{{.*}}'

// ERR-MEDIUM: error: unsupported argument 'medium' to option '-mcmodel=' for target '{{.*}}'
// ERR-KERNEL: error: unsupported argument 'kernel' to option '-mcmodel=' for target '{{.*}}'
// ERR-LARGE:  error: unsupported argument 'large' to option '-mcmodel=' for target '{{.*}}'

// AARCH64-PIC-LARGE: error: invalid argument '-mcmodel=large' only allowed with '-fno-pic'
// ERR-AARCH64_32: error: unsupported argument 'small' to option '-mcmodel=' for target 'aarch64_32-unknown-linux'
