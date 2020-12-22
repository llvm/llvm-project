// Check the -mharden-sls= option, which has a required argument to select
// scope.
// RUN: %clang -target aarch64--none-eabi -c %s -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-OFF --check-prefix=BLR-OFF
// RUN: %clang -target armv7a--none-eabi -c %s -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-OFF --check-prefix=BLR-OFF

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=none 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-OFF --check-prefix=BLR-OFF
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=none 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-OFF --check-prefix=BLR-OFF

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=retbr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-OFF
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=retbr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-OFF

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-OFF --check-prefix=BLR-ON
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-OFF --check-prefix=BLR-ON

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=blr -mharden-sls=none 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-OFF --check-prefix=BLR-OFF
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=blr -mharden-sls=none 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-OFF --check-prefix=BLR-OFF

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=blr -mharden-sls=retbr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-OFF
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=blr -mharden-sls=retbr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-OFF

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=retbr,blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-ON
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=retbr,blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-ON

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=all 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-ON
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=all 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-ON

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=retbr,blr,retbr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-ON
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=retbr,blr,retbr 2>&1 | \
// RUN: FileCheck %s --check-prefix=RETBR-ON --check-prefix=BLR-ON

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=retbr,blr,r 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-SLS-SPEC
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=retbr,blr,r 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-SLS-SPEC

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=none,blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-SLS-SPEC
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=none,blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-SLS-SPEC

// RUN: %clang -target aarch64--none-eabi -c %s -### -mharden-sls=all,-blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-SLS-SPEC
// RUN: %clang -target armv7a--none-eabi -c %s -### -mharden-sls=all,-blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-SLS-SPEC

// RETBR-OFF-NOT: "harden-sls-retbr"
// RETBR-ON:  "+harden-sls-retbr"

// BLR-OFF-NOT: "harden-sls-blr"
// BLR-ON:  "+harden-sls-blr"

// BAD-SLS-SPEC: invalid sls hardening option '{{[^']+}}' in '-mharden-sls=

// RUN: %clang -target armv6a--none-eabi -c %s -### -mharden-sls=all 2>&1 | \
// RUN: FileCheck %s --check-prefix=SLS-NOT-SUPPORTED

// RUN: %clang -target armv6a--none-eabi -c %s -### -mharden-sls=retbr 2>&1 | \
// RUN: FileCheck %s --check-prefix=SLS-NOT-SUPPORTED

// RUN: %clang -target armv6a--none-eabi -c %s -### -mharden-sls=blr 2>&1 | \
// RUN: FileCheck %s --check-prefix=SLS-NOT-SUPPORTED

// RUN: %clang -target armv7r--none-eabi -c %s -### -mharden-sls=all 2>&1 | \
// RUN: FileCheck %s --check-prefix=SLS-NOT-SUPPORTED

// RUN: %clang -target armv7m--none-eabi -c %s -### -mharden-sls=all 2>&1 | \
// RUN: FileCheck %s --check-prefix=SLS-NOT-SUPPORTED

// RUN: %clang -target armv6a--none-eabi -c %s -### -mharden-sls=none 2>&1 | \
// RUN: FileCheck %s --check-prefix=SLS-SUPPORTED

// RUN: %clang -target armv7a-linux-gnueabihf -c %s -### -mharden-sls=all 2>&1 | \
// RUN: FileCheck %s --check-prefix=SLS-SUPPORTED

// RUN: %clang -target armv8a-linux-gnueabihf -c %s -### -mharden-sls=all 2>&1 | \
// RUN: FileCheck %s --check-prefix=SLS-SUPPORTED

// SLS-NOT-SUPPORTED: -mharden-sls is only supported on armv7-a or later
// SLS-SUPPORTED-NOT: mharden-sls

