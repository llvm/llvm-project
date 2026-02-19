

// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-strict-align 2>&1 | FileCheck %s -check-prefixes=ZILSD-4BYTE-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mstrict-align 2>&1 | FileCheck %s -check-prefixes=NO-ZILSD-4BYTE-ALIGN

// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-scalar-strict-align 2>&1 | FileCheck %s -check-prefix=ZILSD-4BYTE-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mscalar-strict-align 2>&1 | FileCheck %s -check-prefix=NO-ZILSD-4BYTE-ALIGN

// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-word-align 2>&1 | FileCheck %s -check-prefix=ZILSD-4BYTE-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-strict-align 2>&1 | FileCheck %s -check-prefix=NO-ZILSD-4BYTE-ALIGN

// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-strict-align -mno-scalar-strict-align 2>&1 | FileCheck %s -check-prefix=ZILSD-4BYTE-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-word-align -mscalar-strict-align 2>&1 | FileCheck %s -check-prefix=NO-ZILSD-4BYTE-ALIGN

// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-strict-align -mno-strict-align 2>&1 | FileCheck %s -check-prefix=ZILSD-4BYTE-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-word-align -mstrict-align 2>&1 | FileCheck %s -check-prefix=NO-ZILSD-4BYTE-ALIGN

// ZILSD-4BYTE-ALIGN: "-target-feature" "+zilsd-4byte-align"
// NO-ZILSD-4BYTE-ALIGN: "-target-feature" "-zilsd-4byte-align"

// RUN: not %clang --target=riscv64-unknown-elf -### %s -mzilsd-word-align 2>&1 | FileCheck %s -check-prefix=ERROR-ZILSD-4BYTE-ALIGN
// RUN: not %clang --target=riscv64-unknown-elf -### %s -mzilsd-strict-align 2>&1 | FileCheck %s -check-prefix=ERROR-NO-ZILSD-4BYTE-ALIGN

// ERROR-ZILSD-4BYTE-ALIGN: error: unsupported option '-mzilsd-word-align' for target
// ERROR-NO-ZILSD-4BYTE-ALIGN: error: unsupported option '-mzilsd-strict-align' for target
