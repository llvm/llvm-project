// Test if PGO instrumentation and use pass are invoked.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=llvm %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN: PGOInstrumentationGenPass
//
// Ensure Pass PGOInstrumentationGenPass is not invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=clang %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG-NOT: PGOInstrumentationGenPass
