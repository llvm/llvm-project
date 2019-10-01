// Test if PGO instrumentation and use pass are invoked.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=llvm %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN --check-prefix=CHECK-INSTRPROF
// RUN: %clang_cc1 -O2 -fprofile-instrument=llvm %s  -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN-NEWPM --check-prefix=CHECK-INSTRPROF-NEWPM
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN: PGOInstrumentationGenPass
// CHECK-INSTRPROF: Frontend instrumentation-based coverage lowering
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN-NEWPM: Running pass: PGOInstrumentationGen on
// CHECK-INSTRPROF-NEWPM: Running pass: InstrProfiling on
//
// Ensure Pass PGOInstrumentationGenPass is not invoked.
// RUN: %clang_cc1 -O2 -fprofile-instrument=clang %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG
// RUN: %clang_cc1 -O2 -fprofile-instrument=clang %s -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG-NEWPM
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG-NOT: PGOInstrumentationGenPass
// CHECK-PGOGENPASS-INVOKED-INSTR-GEN-CLANG-NEWPM-NOT: Running pass: PGOInstrumentationGen on

// RUN: %clang_cc1 -O2 -fprofile-instrument=clang %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-CLANG-INSTRPROF
// RUN: %clang_cc1 -O2 -fprofile-instrument=clang %s -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-CLANG-INSTRPROF-NEWPM
// RUN: %clang_cc1 -O0 -fprofile-instrument=clang %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-CLANG-INSTRPROF
// RUN: %clang_cc1 -O0 -fprofile-instrument=clang %s -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-CLANG-INSTRPROF-NEWPM
// CHECK-CLANG-INSTRPROF: Frontend instrumentation-based coverage lowering
// CHECK-CLANG-INSTRPROF-NEWPM: Running pass: InstrProfiling on

// Ensure Pass PGOInstrumentationUsePass is invoked.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgotestir.profraw
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.profdata %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-INSTR-USE
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.profdata %s -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-INSTR-USE-NEWPM
// CHECK-PGOUSEPASS-INVOKED-INSTR-USE: PGOInstrumentationUsePass
// CHECK-PGOUSEPASS-INVOKED-INSTR-USE-NEWPM: Running pass: PGOInstrumentationUse on
//
// Ensure Pass PGOInstrumentationUsePass is not invoked.
// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/pgotestclang.profraw
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.profdata %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE-CLANG
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.profdata %s -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-USE-CLANG-NEWPM
// CHECK-PGOUSEPASS-INVOKED-USE-CLANG-NOT: PGOInstrumentationUsePass
// CHECK-PGOUSEPASS-INVOKED-USE-CLANG-NEWPM-NOT: Running pass: PGOInstrumentationUse on
