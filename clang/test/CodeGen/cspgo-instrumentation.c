// Test if CSPGO instrumentation and use pass are invoked.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN: Running pass: PGOInstrumentationGenCreateVar on
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN: Running pass: PGOInstrumentationGen on
//
// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/noncs.profdata %S/Inputs/pgotestir.proftext
//
// Ensure Pass PGOInstrumentationUsePass and PGOInstrumentationGenPass are invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use=llvm -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  %s -fdebug-pass-manager  -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2: Running pass: PGOInstrumentationGenCreateVar on
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN2: Running pass: PGOInstrumentationGen on

// Ensure Pass PGOInstrumentationUsePass is invoked only once.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use=llvm -fprofile-instrument-use-path=%t/noncs.profdata %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE
// CHECK-PGOUSEPASS-INVOKED-USE: Running pass: PGOInstrumentationUse
// CHECK-PGOUSEPASS-INVOKED-USE-NOT: Running pass: PGOInstrumentationGenCreateVar
// CHECK-PGOUSEPASS-INVOKED-USE-NOT: Running pass: PGOInstrumentationUse
//
// Ensure Pass PGOInstrumentationUsePass is invoked twice.
// RUN: llvm-profdata merge -o %t/cs.profdata %S/Inputs/pgotestir_cs.proftext
// RUN: %clang_cc1 -O2 -fprofile-instrument-use=csllvm -fprofile-instrument-use-path=%t/cs.profdata %s -fdebug-pass-manager  -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE2
// CHECK-PGOUSEPASS-INVOKED-USE2: Running pass: PGOInstrumentationUse
// CHECK-PGOUSEPASS-INVOKED-USE2: Running pass: PGOInstrumentationUse
