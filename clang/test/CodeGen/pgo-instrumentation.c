// Test if PGO instrumentation and use pass are invoked.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=llvm %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN: PGOInstrumentationGenPass
//
// Ensure Pass PGOInstrumentationGenPass is not invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=clang %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG-NOT: PGOInstrumentationGenPass

// Ensure Pass PGOInstrumentationUsePass is invoked.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgotestir.profraw
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.profdata %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-INSTR-USE
// CHECK-PGOUSEPASS-INVOKED-INSTR-USE: PGOInstrumentationUsePass
//
// Ensure Pass PGOInstrumentationUsePass is not invoked.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgotestclang.profraw
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.profdata %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE-CLANG
// CHECK-PGOUSEPASS-INVOKED-USE-CLANG-NOT: PGOInstrumentationUsePass

