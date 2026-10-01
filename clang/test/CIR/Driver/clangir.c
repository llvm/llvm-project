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

// -fclangir and -fno-clangir are last-wins, in either order.

// RUN: %clang -### -fclangir -fno-clangir -S %s 2>&1 | FileCheck %s --check-prefix=NEG
// RUN: %clang -### -fno-clangir -S %s 2>&1 | FileCheck %s --check-prefix=NEG
// NEG: "-cc1"
// NEG-NOT: "-fclangir"

// RUN: %clang -### -fno-clangir -fclangir -S %s 2>&1 | FileCheck %s --check-prefix=POS
// POS: "-cc1"
// POS-SAME: "-fclangir"

// The frontend must honor the negation too, not just the driver. -fopenacc
// warns only when the CIR pipeline is off, which makes UseClangIRPipeline
// observable at -cc1.

// RUN: %clang_cc1 -fopenacc -fclangir -fno-clangir -emit-llvm-only %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1-OFF
// CC1-OFF: use -fclangir to enable runtime effect

// RUN: %clang_cc1 -fopenacc -fno-clangir -fclangir -emit-llvm-only %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1-ON --allow-empty
// CC1-ON-NOT: use -fclangir to enable runtime effect

void foo() {}
