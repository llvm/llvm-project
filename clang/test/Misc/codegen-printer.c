// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -O3 -S -mllvm -print-after-all \
// RUN:   -mllvm -force-new-pm-codegen -o %t 2>&1 | FileCheck %s

// CHECK: *** IR Dump After CodeGenPreparePass
// CHECK: *** IR Dump After X86AsmPrinterPas

void foo() {}
