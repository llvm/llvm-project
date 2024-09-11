// RUN: %clang --target=riscv32 -### -c -mcmodel=small %s 2>&1 | FileCheck --check-prefix=SMALL %s
// RUN: %clang --target=riscv64 -### -c -mcmodel=small %s 2>&1 | FileCheck --check-prefix=SMALL %s

// RUN: %clang --target=riscv32 -### -c -mcmodel=medlow %s 2>&1 | FileCheck --check-prefix=SMALL %s
// RUN: %clang --target=riscv64 -### -c -mcmodel=medlow %s 2>&1 | FileCheck --check-prefix=SMALL %s

// RUN: %clang --target=riscv32 -### -c -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=MEDIUM %s
// RUN: %clang --target=riscv64 -### -c -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=MEDIUM %s

// RUN: %clang --target=riscv32 -### -c -mcmodel=medany %s 2>&1 | FileCheck --check-prefix=MEDIUM %s
// RUN: %clang --target=riscv64 -### -c -mcmodel=medany %s 2>&1 | FileCheck --check-prefix=MEDIUM %s

// SMALL: "-mcmodel=small"
// MEDIUM: "-mcmodel=medium"
