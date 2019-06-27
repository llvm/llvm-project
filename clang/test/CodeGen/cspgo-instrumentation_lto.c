// Test if CSPGO instrumentation and use pass are invoked in lto.
//
// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/noncs.profdata %S/Inputs/pgotestir.proftext
//
// Ensure Pass PGOInstrumentationGenPass is not invoked in PreLink.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm %s -flto -mllvm -debug-pass=Structure -emit-llvm-bc -o %t/foo_fe.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm %s -flto -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm-bc -o %t/foo_fe_pm.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NEWPM
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE: PGOInstrumentationUsePass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE: PGOInstrumentationGenCreateVarPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NOT: PGOInstrumentationGenPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NEWPM: Running pass: PGOInstrumentationGenCreateVar
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NEWPM-NOT: Running pass: PGOInstrumentationGen on
//
// Ensure Pass PGOInstrumentationGenPass is invoked in PostLink.
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe.bc -fprofile-instrument=csllvm -emit-llvm -mllvm -debug-pass=Structure -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fexperimental-new-pass-manager -fdebug-pass-manager -fprofile-instrument=csllvm -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NEWPM
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NOT: PGOInstrumentationUsePass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST: PGOInstrumentationGenPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NOT: PGOInstrumentationGenCreateVarPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NEWPM-NOT: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NEWPM: Running pass: PGOInstrumentationGen on
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NEWPM-NOT: Running pass: PGOInstrumentationGenCreateVar
//
// RUN: llvm-profdata merge -o %t/cs.profdata %S/Inputs/pgotestir_cs.proftext
//
// Ensure Pass PGOInstrumentationUsePass is invoked Once in PreLink.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/cs.profdata %s -flto -mllvm -debug-pass=Structure -emit-llvm-bc -o %t/foo_fe.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/cs.profdata %s -flto -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm-bc -o %t/foo_fe_pm.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NEWPM
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE: PGOInstrumentationUsePass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NOT: PGOInstrumentationGenCreateVarPass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NOT: PGOInstrumentationUsePass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NEWPM-NOT: Running pass: PGOInstrumentationGenCreateVar
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NEWPM-NOT: Running pass: PGOInstrumentationUse
//
// Ensure Pass PGOInstrumentationUSEPass is invoked in PostLink.
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe.bc -fprofile-instrument-use-path=%t/cs.profdata -flto -emit-llvm -mllvm -debug-pass=Structure -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fexperimental-new-pass-manager -fdebug-pass-manager -fprofile-instrument-use-path=%t/cs.profdata -flto -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NEWPM
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST: PGOInstrumentationUsePass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NOT: PGOInstrumentationUsePass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NEWPM-NOT: Running pass: PGOInstrumentationUse
