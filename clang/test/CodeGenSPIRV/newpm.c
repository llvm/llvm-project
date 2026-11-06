// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -triple spirv64 -S -mllvm -print-after=spirv-asm-printer \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NPM
// RUN: %clang_cc1 -triple spirv64 -S -mllvm -print-after=spirv-asm-printer \
// RUN:   -mllvm -force-new-pm-codegen=false %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LEGACY

// NPM: *** IR Dump After SPIRVAsmPrinterPass on foo ***
// LEGACY: *** IR Dump After SPIRV Assembly Printer (spirv-asm-printer) ***

void foo(void) {}
