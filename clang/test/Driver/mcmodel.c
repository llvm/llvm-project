// RUN: %clang --target=x86_64 -### -c -mcmodel=tiny %s 2>&1 | FileCheck --check-prefix=TINY %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=small %s 2>&1 | FileCheck --check-prefix=SMALL %s
// RUN: %clang --target=x86_64 -### -S -mcmodel=kernel %s 2>&1 | FileCheck --check-prefix=KERNEL %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=MEDIUM %s
// RUN: %clang --target=x86_64 -### -S -mcmodel=large %s 2>&1 | FileCheck --check-prefix=LARGE %s
// RUN: %clang --target=powerpc-unknown-aix -### -S -mcmodel=medium %s 2> %t.log
// RUN: FileCheck --check-prefix=AIX-MCMEDIUM-OVERRIDE %s < %t.log
// RUN: not %clang -c -mcmodel=lager %s 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not %clang -c --target=aarch64 -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=AARCH64-MEDIUM %s
// RUN: not %clang -c --target=aarch64 -mcmodel=kernel %s 2>&1 | FileCheck --check-prefix=AARCH64-KERNEL %s

// TINY: "-mcmodel=tiny"
// SMALL: "-mcmodel=small"
// KERNEL: "-mcmodel=kernel"
// MEDIUM: "-mcmodel=medium"
// LARGE: "-mcmodel=large"
// AIX-MCMEDIUM-OVERRIDE: "-mcmodel=large"

// INVALID: error: invalid argument 'lager' to -mcmodel=

// AARCH64-MEDIUM: error: invalid argument 'medium' to -mcmodel=
// AARCH64-KERNEL: error: invalid argument 'kernel' to -mcmodel=
