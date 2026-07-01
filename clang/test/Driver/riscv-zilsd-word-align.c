// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-strict-align 2>&1 | FileCheck %s -check-prefixes=ZILSD-WORD-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mstrict-align 2>&1 | FileCheck %s -check-prefixes=NO-ZILSD-WORD-ALIGN

// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-scalar-strict-align 2>&1 | FileCheck %s -check-prefix=ZILSD-WORD-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mscalar-strict-align 2>&1 | FileCheck %s -check-prefix=NO-ZILSD-WORD-ALIGN

// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-word-align 2>&1 | FileCheck %s -check-prefix=ZILSD-WORD-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-strict-align 2>&1 | FileCheck %s -check-prefix=NO-ZILSD-WORD-ALIGN

// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-strict-align -mno-scalar-strict-align 2>&1 | FileCheck %s -check-prefix=ZILSD-WORD-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-word-align -mscalar-strict-align 2>&1 | FileCheck %s -check-prefix=NO-ZILSD-WORD-ALIGN

// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-strict-align -mno-strict-align 2>&1 | FileCheck %s -check-prefix=ZILSD-WORD-ALIGN
// RUN: %clang --target=riscv32-unknown-elf -### %s -mzilsd-word-align -mstrict-align 2>&1 | FileCheck %s -check-prefix=NO-ZILSD-WORD-ALIGN

// ZILSD-WORD-ALIGN: "-target-feature" "+zilsd-word-align"
// NO-ZILSD-WORD-ALIGN: "-target-feature" "-zilsd-word-align"

// RUN: not %clang --target=riscv64-unknown-elf -### %s -mzilsd-word-align 2>&1 | FileCheck %s -check-prefix=ERROR-ZILSD-WORD-ALIGN
// RUN: not %clang --target=riscv64-unknown-elf -### %s -mzilsd-strict-align 2>&1 | FileCheck %s -check-prefix=ERROR-NO-ZILSD-WORD-ALIGN

// ERROR-ZILSD-WORD-ALIGN: error: unsupported option '-mzilsd-word-align' for target
// ERROR-NO-ZILSD-WORD-ALIGN: error: unsupported option '-mzilsd-strict-align' for target
