// Test if PGO instrumentation and use pass are invoked.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=llvm %s  -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN --check-prefix=CHECK-INSTRPROF
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN: Running pass: PGOInstrumentationGen on
// CHECK-INSTRPROF: Running pass: InstrProfiling on
//
// Ensure Pass PGOInstrumentationGenPass is not invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=clang %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG-NOT: Running pass: PGOInstrumentationGen on

// RUN: %clang_cc1 -O2 -fprofile-instrument=clang %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-CLANG-INSTRPROF
// RUN: %clang_cc1 -O0 -fprofile-instrument=clang %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-CLANG-INSTRPROF
// CHECK-CLANG-INSTRPROF: Running pass: InstrProfiling on

// Ensure Pass PGOInstrumentationUsePass is invoked.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgotestir.profraw
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.profdata %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-INSTR-USE
// CHECK-PGOUSEPASS-INVOKED-INSTR-USE: Running pass: PGOInstrumentationUse on
//
// Ensure Pass PGOInstrumentationUsePass is not invoked.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgotestclang.profraw
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.profdata %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE-CLANG
// CHECK-PGOUSEPASS-INVOKED-USE-CLANG-NOT: Running pass: PGOInstrumentationUse on
