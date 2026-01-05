// This file tests the -mtp=<mode> functionality in Clang’s ARM driver.
// It verifies:
//
//   1. ARMv7 targets: explicit hardware modes, explicit soft mode, and auto mode.
//   2. M Profile variants: explicit hardware mode should fail and auto mode defaults to soft.
//   3. ARMv6 variants: explicit hardware modes on ARMv6K/KZ work, but auto mode falls back to soft when Thumb2 is missing.
//   4. ARMv5 variants: explicit hardware mode is rejected and auto mode defaults to soft.
//   5. Miscellaneous error cases (e.g. empty -mtp value).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 1. ARMv7 Targets
//===----------------------------------------------------------------------===//

// Test explicit hardware mode using "tpidrprw" on an ARMv7 target.
// RUN: %clang --target=armv7-linux -mtp=tpidrprw -### -S %s 2>&1 | FileCheck -check-prefix=ARMv7_TPIDRPRW %s
// ARMv7_TPIDRPRW: "-target-feature" "+read-tp-tpidrprw"

// Test explicit hardware mode using "tpidrurw" on an ARMv7 target.
// RUN: %clang --target=armv7-linux -mtp=tpidrurw -### -S %s 2>&1 | FileCheck -check-prefix=ARMv7_TPIDRURW %s
// ARMv7_TPIDRURW: "-target-feature" "+read-tp-tpidrurw"

// Test explicit hardware mode using "tpidruro" on an ARMv7 target.
// RUN: %clang --target=armv7-linux -mtp=tpidruro -### -S %s 2>&1 | FileCheck -check-prefix=ARMv7_TPIDRURO %s
// ARMv7_TPIDRURO: "-target-feature" "+read-tp-tpidruro"

// Test explicit "soft" mode on an ARMv7 target (forces software mode).
// RUN: %clang --target=armv7-linux -mtp=soft -### -S %s 2>&1 | FileCheck -check-prefix=ARM_Soft %s
// ARM_Soft-NOT: "-target-feature" "+read-tp-"

// Test auto mode on an ARMv7 target (hardware support and Thumb2 yield HW mode).
// RUN: %clang --target=armv7-linux -mtp=auto -### -S %s 2>&1 | FileCheck -check-prefix=ARMv7_Auto %s
// Default mode is implicitly -mtp=auto
// RUN: %clang --target=armv7-linux -### -S %s 2>&1 | FileCheck -check-prefix=ARMv7_Auto %s
// ARMv7_Auto: "-target-feature" "+read-tp-tpidruro"

//===----------------------------------------------------------------------===//
// 2. M Profile Variants (e.g. thumbv6t2)
//===----------------------------------------------------------------------===//

// Test explicit hardware mode on a M Profile target: thumbv6t2 does not support CP15.
// RUN: not %clang --target=thumbv6t2-linux -mtp=cp15 -### -S %s 2>&1 | FileCheck -check-prefix=Thumbv6t2_Error %s
// Thumbv6t2_Error: error: hardware TLS register is not supported for the armv6t2 sub-architecture

// Test auto mode on a M Profile target: should default to soft mode.
// RUN: %clang --target=thumbv6t2-linux -mtp=auto -### -S %s 2>&1 | FileCheck -check-prefix=Thumbv6t2_Auto %s
// Thumbv6t2_Auto-NOT: "-target-feature" "+read-tp-"


//===----------------------------------------------------------------------===//
// 3. ARMv6 Variants
//===----------------------------------------------------------------------===//

// Test explicit hardware mode using "cp15" on an ARMv6K and ARMv6KZ targets.
// RUN: %clang --target=armv6k-linux -mtp=cp15 -### -S %s 2>&1 | FileCheck -check-prefix=ARMv6k_Cp15 %s
// RUN: %clang --target=armv6kz-linux -mtp=cp15 -### -S %s 2>&1 | FileCheck -check-prefix=ARMv6k_Cp15 %s
// ARMv6k_Cp15: "-target-feature" "+read-tp-tpidruro"


// Test auto mode on ARMv6K and ARMv6KZ targets: defaults to soft mode due to missing Thumb2 encoding.
// RUN: %clang --target=armv6k-linux -mtp=auto -### -S %s 2>&1 | FileCheck -check-prefix=ARMv6k_Auto %s
// RUN: %clang --target=armv6kz-linux -mtp=auto -### -S %s 2>&1 | FileCheck -check-prefix=ARMv6k_Auto %s
// ARMv6k_Auto-NOT: "-target-feature" "+read-tp-"


//===----------------------------------------------------------------------===//
// 4. ARMv5 Variants
//===----------------------------------------------------------------------===//

// Test explicit hardware mode on an ARMv5T target: hardware TP is not supported.
// RUN: not %clang --target=armv5t-linux -mtp=cp15 -### -S %s 2>&1 | FileCheck -check-prefix=ARMv5t_Error %s
// ARMv5t_Error: error: hardware TLS register is not supported for the armv5 sub-architecture

// Test auto mode on an ARMv5T target: should default to soft mode.
// RUN: %clang --target=armv5t-linux -mtp=auto -### -S %s 2>&1 | FileCheck -check-prefix=ARMv5t_Auto %s
// ARMv5t_Auto-NOT: "-target-feature" "+read-tp-"

//===----------------------------------------------------------------------===//
// 5. Miscellaneous Tests
//===----------------------------------------------------------------------===//

// Test empty -mtp value on an ARMv7 target: should produce a missing argument error.
// RUN: not %clang --target=armv7-linux -mtp= -### -S %s 2>&1 | FileCheck -check-prefix=Empty_MTP %s
// Empty_MTP: error: {{.*}}missing

// Test explicit hardware mode in assembler mode on an unsupporting target does not fail with error
// RUN: %clang --target=thumbv6t2-linux -mtp=cp15 -x assembler -### %s 2>&1 | FileCheck -check-prefix=Thumbv6t2_Asm %s
// Thumbv6t2_Asm-NOT: "-target-feature" "+read-tp-"

