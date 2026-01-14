// Test if CSPGO instrumentation and use pass are invoked in lto.
//
// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/noncs.profdata %S/Inputs/pgotestir.proftext
//
// Ensure Pass PGOInstrumentationGenPass is not invoked in PreLink.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use=llvm -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm %s -flto -fdebug-pass-manager -emit-llvm-bc -o %t/foo_fe_pm.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE: Running pass: PGOInstrumentationGenCreateVar
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NOT: Running pass: PGOInstrumentationGen on
//
// Ensure Pass PGOInstrumentationGenPass is invoked in PostLink.
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fdebug-pass-manager -fprofile-instrument=csllvm -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NOT: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST: Running pass: PGOInstrumentationGen on
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NOT: Running pass: PGOInstrumentationGenCreateVar
//
// RUN: llvm-profdata merge -o %t/cs.profdata %S/Inputs/pgotestir_cs.proftext
//
// Ensure Pass PGOInstrumentationUsePass is invoked Once in PreLink.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use=csllvm -fprofile-instrument-use-path=%t/cs.profdata %s -flto -fdebug-pass-manager -emit-llvm-bc -o %t/foo_fe_pm.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE: Running pass: PGOInstrumentationUse
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NOT: Running pass: PGOInstrumentationGenCreateVar
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NOT: Running pass: PGOInstrumentationUse
//
// Ensure Pass PGOInstrumentationUSEPass is invoked in PostLink.
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fdebug-pass-manager -fprofile-instrument-use=csllvm -fprofile-instrument-use-path=%t/cs.profdata -flto -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST: Running pass: PGOInstrumentationUse
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NOT: Running pass: PGOInstrumentationUse
