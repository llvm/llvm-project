// Test if CSPGO instrumentation and use pass are invoked in thinlto.
//
// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/noncs.profdata %S/Inputs/pgotestir.proftext
//
// Ensure Pass PGOInstrumentationGenPass is not invoked in PreLink.
// RUN: %clang_cc1 -O2 -fno-experimental-new-pass-manager -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm %s -fprofile-instrument-path=default.profraw -flto=thin -mllvm -debug-pass=Structure -emit-llvm-bc -o %t/foo_fe.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/noncs.profdata -fprofile-instrument=csllvm %s -fprofile-instrument-path=default.profraw  -flto=thin -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm-bc -o %t/foo_fe_pm.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NEWPM
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE: PGOInstrumentationUsePass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE: PGOInstrumentationGenCreateVarPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NOT: PGOInstrumentationGenPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NEWPM: Running pass: PGOInstrumentationGenCreateVar
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-PRE-NEWPM-NOT: Running pass: PGOInstrumentationGen on
//
// RUN: llvm-lto -thinlto -o %t/foo %t/foo_fe.bc
// RUN: llvm-lto -thinlto -o %t/foo_pm %t/foo_fe_pm.bc
// Ensure Pass PGOInstrumentationGenPass is invoked in PostLink.
// RUN: %clang_cc1 -O2 -fno-experimental-new-pass-manager -x ir %t/foo_fe.bc -fthinlto-index=%t/foo.thinlto.bc -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  -flto=thin -emit-llvm -mllvm -debug-pass=Structure -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fthinlto-index=%t/foo_pm.thinlto.bc -fexperimental-new-pass-manager -fdebug-pass-manager  -fprofile-instrument=csllvm -fprofile-instrument-path=default.profraw  -flto=thin -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NEWPM
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NOT: PGOInstrumentationUsePass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NOT: PGOInstrumentationGenCreateVarPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST: PGOInstrumentationGenPass
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NEWPM-NOT: Running pass: PGOInstrumentationUse
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NEWPM-NOT: Running pass: PGOInstrumentationGenCreateVar
// CHECK-CSPGOGENPASS-INVOKED-INSTR-GEN-POST-NEWPM: Running pass: PGOInstrumentationGen on
//
// RUN: llvm-profdata merge -o %t/cs.profdata %S/Inputs/pgotestir_cs.proftext
//
// Ensure Pass PGOInstrumentationUsePass is invoked Once in PreLink.
// RUN: %clang_cc1 -O2 -fno-experimental-new-pass-manager -fprofile-instrument-use-path=%t/cs.profdata %s -flto=thin -mllvm -debug-pass=Structure -emit-llvm-bc -o %t/foo_fe.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t/cs.profdata %s -flto=thin -fexperimental-new-pass-manager -fdebug-pass-manager -emit-llvm-bc -o %t/foo_fe_pm.bc 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NEWPM
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE: PGOInstrumentationUsePass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NOT: PGOInstrumentationUsePass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-PRE-NEWPM-NOT: Running pass: PGOInstrumentationUse
//
// RUN: llvm-lto -thinlto -o %t/foo %t/foo_fe.bc
// RUN: llvm-lto -thinlto -o %t/foo_pm %t/foo_fe_pm.bc
// Ensure Pass PGOInstrumentationUSEPass is invoked in PostLink.
// RUN: %clang_cc1 -O2 -fno-experimental-new-pass-manager -x ir %t/foo_fe.bc -fthinlto-index=%t/foo.thinlto.bc -fprofile-instrument-use-path=%t/cs.profdata -flto=thin -emit-llvm -mllvm -debug-pass=Structure -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST -dump-input=always
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fthinlto-index=%t/foo_pm.thinlto.bc -fexperimental-new-pass-manager -fdebug-pass-manager -fprofile-instrument-use-path=%t/cs.profdata -flto=thin -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NEWPM -dump-input=always
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST: PGOInstrumentationUsePass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NOT: PGOInstrumentationUsePass
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NEWPM: Running pass: PGOInstrumentationUse
// CHECK-CSPGOUSEPASS-INVOKED-INSTR-USE-POST-NEWPM-NOT: Running pass: PGOInstrumentationUse
//
// Finally, test if a non-cs profile is passed to PostLink passes, PGO UsePass is not invoked.
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe.bc -fthinlto-index=%t/foo.thinlto.bc -fprofile-instrument-use-path=%t/noncs.profdata -flto=thin -emit-llvm -mllvm -debug-pass=Structure -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-INSTR-USE-POST
// RUN: %clang_cc1 -O2 -x ir %t/foo_fe_pm.bc -fthinlto-index=%t/foo_pm.thinlto.bc -fexperimental-new-pass-manager -fdebug-pass-manager -fprofile-instrument-use-path=%t/noncs.profdata -flto=thin -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PGOUSEPASS-INVOKED-INSTR-USE-POST-NEWPM
// CHECK-PGOUSEPASS-INVOKED-INSTR-USE-POST-NOT: PGOInstrumentationUsePass
// CHECK-PGOUSEPASS-INVOKED-INSTR-USE-POST-NEWPM-NOT: Running pass: PGOInstrumentationUse
