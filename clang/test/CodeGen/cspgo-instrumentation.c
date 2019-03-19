// Test if CSPGO instrumentation and use pass are invoked.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN
// RUN: %clang_cc1 -O2 -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  %s -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-NEWPM
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN: PGOInstrumentationGenCreateVarPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN: PGOInstrumentationGenPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-NEWPM: Running pass: PGOInstrumentationGenCreateVar on
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-NEWPM: Running pass: PGOInstrumentationGen on
//
// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/noncs.profdata %S/Inputs/pgotestir.proftext
//
// Ensure Pass PGOInstrumentationUsePass and PGOInstrumentationGenPass are invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  %s -fexperimental-new-pass-manager -fdebug-pass-manager  -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2-NEWPM
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2: PGOInstrumentationUsePass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2: PGOInstrumentationGenCreateVarPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2: PGOInstrumentationGenPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2-NEWPM: Running pass: PGOInstrumentationGenCreateVar on
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2-NEWPM: Running pass: PGOInstrumentationGen on

// Ensure Pass PGOInstrumentationUsePass is invoked only once.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/noncs.profdata %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/noncs.profdata %s -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE-NEWPM
// CHECK-PGOUSEPASS-INVOKED-USE: PGOInstrumentationUsePass
// CHECK-PGOUSEPASS-INVOKED-USE-NOT: PGOInstrumentationGenCreateVarPass
// CHECK-PGOUSEPASS-INVOKED-USE-NOT: PGOInstrumentationUsePass
// CHECK-PGOUSEPASS-INVOKED-USE-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-PGOUSEPASS-INVOKED-USE-NEWPM-NOT: Running pass: PGOInstrumentationGenCreateVar
// CHECK-PGOUSEPASS-INVOKED-USE-NEWPM-NOT: Running pass: PGOInstrumentationUse
//
// Ensure Pass PGOInstrumentationUsePass is invoked twice.
// RUN: llvm-profdata merge -o %t/cs.profdata %S/Inputs/pgotestir_cs.proftext
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/cs.profdata %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE2
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/cs.profdata %s -fexperimental-new-pass-manager -fdebug-pass-manager  -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE2-NEWPM
// CHECK-PGOUSEPASS-INVOKED-USE2: PGOInstrumentationUsePass
// CHECK-PGOUSEPASS-INVOKED-USE2: PGOInstrumentationUsePass
// CHECK-PGOUSEPASS-INVOKED-USE2-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-PGOUSEPASS-INVOKED-USE2-NEWPM: Running pass: PGOInstrumentationUse
