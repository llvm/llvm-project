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

// RUN: %clang --target=armv6k-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// ARM_THREAD_POINTER-HARD: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv6k-linux -mtp=auto -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER_AUTO %s
// ARM_THREAD_POINTER_AUTO-NOT: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=thumbv6k-apple-darwin -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=THUMBv6_THREAD_POINTER_NO_AUTO %s
// THUMBv6_THREAD_POINTER_NO_AUTO-NOT: "-target-feature" "+read-tp-tpidruro"

// RUN: not %clang --target=thumbv6k-apple-darwin -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=THUMBv6_THREAD_POINTER_NO_HARD %s
// THUMBv6_THREAD_POINTER_NO_HARD: unsupported option '-mtp=' for target 'thumbv6k-apple-darwin'

// RUN: not %clang --target=thumbv6t2-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER_NO_HARD %s
// ARM_THREAD_POINTER_NO_HARD: hardware TLS register is not supported for the armv6t2 sub-architecture

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
// ARMv7_THREAD_POINTER_NON: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv7-linux -mtp=auto -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER_Auto %s
// ARMv7_THREAD_POINTER_Auto: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv7-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER_HARD %s
// ARMv7_THREAD_POINTER_HARD: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv7m-linux -mtp=auto -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7m_THREAD_POINTER_Auto %s
// ARMv7m_THREAD_POINTER_Auto-NOT: "-target-feature" "+read-tp-tpidruro"

// RUN: not %clang --target=armv7m-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7m_THREAD_POINTER_HARD %s
// ARMv7m_THREAD_POINTER_HARD: hardware TLS register is not supported for the thumbv7m sub-architecture

// RUN: %clang --target=armv5t-linux -mtp=auto -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv5t_THREAD_POINTER_Auto %s
// ARMv5t_THREAD_POINTER_Auto-NOT: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv6k-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv6k_THREAD_POINTER_Auto %s
// ARMv6k_THREAD_POINTER_Auto: "-target-feature" "+read-tp-tpidruro"

// RUN: not %clang --target=armv6t2-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv6t2_THREAD_POINTER_HARD %s
// ARMv6t2_THREAD_POINTER_HARD: hardware TLS register is not supported for the armv6t2 sub-architecture

// RUN: %clang --target=armv6t2-linux -mtp=auto -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMV6t2_THREAD_POINTER_AUTO %s
// ARMV6t2_THREAD_POINTER_AUTO-NOT: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv6kz-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv6kz_THREAD_POINTER_HARD %s
// ARMv6kz_THREAD_POINTER_HARD: "-target-feature" "+read-tp-tpidruro"

// RUN: %clang --target=armv6kz-linux -mtp=auto -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMV6KZ_THREAD_POINTER_AUTO %s
// ARMV6KZ_THREAD_POINTER_AUTO-NOT: "-target-feature" "+read-tp-tpidruro"