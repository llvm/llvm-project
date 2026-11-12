// Test of the AArch64 values of -mtp=, checking that each one maps to
// the right target features.

// RUN: %clang --target=aarch64-linux -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL0 %s
// RUN: %clang --target=aarch64-linux -### -S %s -mtp=el0 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL0 %s
// RUN: %clang --target=aarch64-linux -### -S %s -mtp=tpidr_el0 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL0 %s
// ARMv8_THREAD_POINTER_EL0-NOT: "-target-feature" "+tpidrro-el0"
// ARMv8_THREAD_POINTER_EL0-NOT: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_EL0-NOT: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_EL0-NOT: "-target-feature" "+tpidr-el3"

// RUN: %clang --target=aarch64-linux -### -S %s -mtp=tpidrro_el0 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_ROEL0 %s
// ARMv8_THREAD_POINTER_ROEL0:     "-target-feature" "+tpidrro-el0"
// ARMv8_THREAD_POINTER_ROEL0-NOT: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_ROEL0-NOT: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_ROEL0-NOT: "-target-feature" "+tpidr-el3"

// RUN: %clang --target=aarch64-linux -### -S %s -mtp=el1 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL1 %s
// RUN: %clang --target=aarch64-linux -### -S %s -mtp=tpidr_el1 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL1 %s
// ARMv8_THREAD_POINTER_EL1-NOT: "-target-feature" "+tpidrro-el0"
// ARMv8_THREAD_POINTER_EL1:     "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_EL1-NOT: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_EL1-NOT: "-target-feature" "+tpidr-el3"

// RUN: %clang --target=aarch64-linux -### -S %s -mtp=el2 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL2 %s
// RUN: %clang --target=aarch64-linux -### -S %s -mtp=tpidr_el2 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL2 %s
// ARMv8_THREAD_POINTER_EL2-NOT: "-target-feature" "+tpidrro-el0"
// ARMv8_THREAD_POINTER_EL2-NOT: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_EL2:     "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_EL2-NOT: "-target-feature" "+tpidr-el3"

// RUN: %clang --target=aarch64-linux -### -S %s -mtp=el3 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL3 %s
// RUN: %clang --target=aarch64-linux -### -S %s -mtp=tpidr_el3 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL3 %s
// ARMv8_THREAD_POINTER_EL3-NOT: "-target-feature" "+tpidrro-el0"
// ARMv8_THREAD_POINTER_EL3-NOT: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_EL3-NOT: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_EL3:     "-target-feature" "+tpidr-el3"
