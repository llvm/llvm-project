// Tests related to -fclangir option.

// Verify that -fclangir is always forwarded to -cc1 by the driver, and
// that the frontend ignores it when the input is LLVM IR.

// -fclangir should be passed to -cc1 for source inputs.
// RUN: %clang -### -fclangir -S %s 2>&1 | FileCheck %s --check-prefix=SOURCE
// SOURCE: "-cc1"
// SOURCE-SAME: "-fclangir"
// SOURCE-SAME: "-x" "c"

// -fclangir should also be passed to -cc1 for LLVM IR inputs (the frontend
// will ignore it and use the standard LLVM backend).
// RUN: %clang -### -fclangir -S -x ir /dev/null 2>&1 | FileCheck %s --check-prefix=LLVMIR
// LLVMIR: "-cc1"
// LLVMIR-SAME: "-fclangir"

void foo() {}
