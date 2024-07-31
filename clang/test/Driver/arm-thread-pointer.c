// Test of the AArch32 values of -mtp=, checking that each one maps to
// the right target features.

// RUN: %clang --target=armv7-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER-HARD %s
// ARMv7_THREAD_POINTER-HARD: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv7-linux -mtp=tpidruro -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER-HARD %s
// RUN: %clang --target=armv7-linux -mtp=tpidrurw -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER-TPIDRURW %s
// ARMv7_THREAD_POINTER-TPIDRURW: "-target-feature" "+read-tp-tpidrurw"
// RUN: %clang --target=armv7-linux -mtp=tpidrprw -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER-TPIDRPRW %s
// ARMv7_THREAD_POINTER-TPIDRPRW: "-target-feature" "+read-tp-tpidrprw"

// RUN: %clang --target=armv6t2-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// RUN: %clang --target=thumbv6t2-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// RUN: %clang --target=armv6k-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// RUN: %clang --target=armv6-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// RUN: %clang --target=armv5t-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// ARM_THREAD_POINTER-HARD: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv5t-linux -mtp=cp15 -x assembler -### %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv5_THREAD_POINTER_ASSEMBLER %s
// ARMv5_THREAD_POINTER_ASSEMBLER-NOT: hardware TLS register is not supported for the armv5 sub-architecture

// RUN: not %clang --target=armv6-linux -mthumb -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=THUMBv6_THREAD_POINTER_UNSUPP %s
// RUN: not %clang --target=thumbv6-linux -mthumb -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=THUMBv6_THREAD_POINTER_UNSUPP %s
// THUMBv6_THREAD_POINTER_UNSUPP: hardware TLS register is not supported for the thumbv6 sub-architecture

// RUN: %clang --target=armv7-linux -mtp=soft -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER_SOFT %s
// ARMv7_THREAD_POINTER_SOFT-NOT: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv7-linux -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER_NON %s
// ARMv7_THREAD_POINTER_NON-NOT: "-target-feature" "+read-tp-tpidruro"
