// REQUIRES: llvm-driver, webassembly-registered-target, crash-recovery

// A recovered compiler crash must not execute the next independent cc1 job.
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t && not %clang -fintegrated-cc1 -fno-crash-diagnostics \
// RUN:   --target=wasm32-unknown-unknown -c before.c crash.c after.c 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not='Error handler already registered'
// RUN: test -f %t/before.o
// RUN: not test -f %t/after.o
// CHECK: fatal error: error in backend: #pragma clang __debug llvm_fatal_error

//--- before.c
int before(void) { return 1; }

//--- crash.c
#pragma clang __debug llvm_fatal_error

//--- after.c
int after(void) { return 2; }
