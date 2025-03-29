// Test if CSPGO instrumentation and use pass are invoked in thinlto.
//
// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/noncs.profdata %S/Inputs/pgotestir.proftext
//
// Ensure Pass PGOInstrumentationGenPass is not invoked in PreLink.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm %s -fprofile-instrument-path=default.profraw  -flto=thin -fdebug-pass-manager -emit-llvm-bc -o %t/foo_fe_pm.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE: Running pass: PGOInstrumentationGenCreateVar
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NOT: Running pass: PGOInstrumentationGen on
//
// RUN: llvm-lto -thinlto -o %t/foo_pm %t/foo_fe_pm.bc
// Ensure Pass PGOInstrumentationGenPass is invoked in PostLink.
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fthinlto-index=%t/foo_pm.thinlto.bc -fdebug-pass-manager  -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  -flto=thin -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NOT: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NOT: Running pass: PGOInstrumentationGenCreateVar
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST: Running pass: PGOInstrumentationGen on
//
// RUN: llvm-profdata merge -o %t/cs.profdata %S/Inputs/pgotestir_cs.proftext
//
// Ensure Pass PGOInstrumentationUsePass is invoked Once in PreLink.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/cs.profdata %s -flto=thin -fdebug-pass-manager -emit-llvm-bc -o %t/foo_fe_pm.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE: Running pass: PGOInstrumentationUse
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NOT: Running pass: PGOInstrumentationUse
//
// RUN: llvm-lto -thinlto -o %t/foo_pm %t/foo_fe_pm.bc
// Ensure Pass PGOInstrumentationUSEPass is invoked in PostLink.
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fthinlto-index=%t/foo_pm.thinlto.bc -fdebug-pass-manager -fprofile-instrument-use-path=%t/cs.profdata -flto=thin -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST: Running pass: PGOInstrumentationUse
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NOT: Running pass: PGOInstrumentationUse
//
// Finally, test if a non-cs profile is passed to PostLink passes, PGO UsePass is not invoked.
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fthinlto-index=%t/foo_pm.thinlto.bc -fdebug-pass-manager -fprofile-instrument-use-path=%t/noncs.profdata -flto=thin -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-INSTR-USE-POST
// CHECK-PGOUSEPASS-INVOKED-INSTR-USE-POST-NOT: Running pass: PGOInstrumentationUse
