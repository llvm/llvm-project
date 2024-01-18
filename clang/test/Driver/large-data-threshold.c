// RUN: %clang --target=x86_64 -### -c -mcmodel=medium -mlarge-data-threshold=200 %s 2>&1 | FileCheck --check-prefix=ARG %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=small -mlarge-data-threshold=200 %s 2>&1 | FileCheck --check-prefix=SMALL %s
// RUN: not %clang --target=riscv32 -### -c -mcmodel=medium -mlarge-data-threshold=200 %s 2>&1 | FileCheck --check-prefix=ARCH %s

// ARG: "-mlarge-data-threshold=200"

// SMALL: 'mlarge-data-threshold=' only applies to medium code model
// ARCH: unsupported option 'mlarge-data-threshold=' for target 'riscv32'
