// RUN: %clang --target=x86_64 -### -c -mcmodel=large -mlarge-data-threshold=200 %s 2>&1 | FileCheck --check-prefix=ARG %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=medium -mlarge-data-threshold=200 %s 2>&1 | FileCheck --check-prefix=ARG %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=large %s 2>&1 | FileCheck --check-prefix=ARG-LARGE-DEFAULT %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=ARG-MEDIUM-DEFAULT %s
// RUN: %clang --target=x86_64 -### -c -mcmodel=small -mlarge-data-threshold=200 %s 2>&1 | FileCheck --check-prefix=SMALL --implicit-check-not=mlarge-data-threshold %s
// RUN: not %clang --target=riscv32 -### -c -mcmodel=medium -mlarge-data-threshold=200 %s 2>&1 | FileCheck --check-prefix=ARCH %s

// ARG: "-mlarge-data-threshold=200"
// ARG-MEDIUM-DEFAULT: "-mlarge-data-threshold=65536"
// ARG-LARGE-DEFAULT: "-mlarge-data-threshold=0"

// SMALL: 'mlarge-data-threshold=' only applies to medium and large code models
// ARCH: unsupported option '-mlarge-data-threshold=' for target
