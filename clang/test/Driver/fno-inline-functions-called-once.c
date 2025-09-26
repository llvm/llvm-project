// REQUIRES: x86-registered-target

// Check that -fno-inline-functions-called-once is forwarded to LLVM.
// RUN: %clang -### -S %s -fno-inline-functions-called-once 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FWD
// FWD: "-mllvm" "-no-inline-functions-called-once"

// Check that the positive form does NOT forward anything to -mllvm.
// RUN: %clang -### -S %s -finline-functions-called-once 2>&1 \
// RUN:   | FileCheck %s --check-prefix=POS
// POS-NOT: -mllvm
// POS-NOT: -no-inline-functions-called-once

// Help text should show both flags (order-independent).
// RUN: %clang --help 2>&1 | FileCheck %s --check-prefix=HELP
// HELP-DAG: -finline-functions-called-once
// HELP-DAG: -fno-inline-functions-called-once

int x;

