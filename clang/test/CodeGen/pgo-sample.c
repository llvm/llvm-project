// Test if PGO sample use passes are invoked.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -O2 -fno-experimental-new-pass-manager -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=LEGACY
// RUN: %clang_cc1 -O2 -fexperimental-new-pass-manager -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=NEWPM

// LEGACY: Remove unused exception handling info
// LEGACY: Sample profile pass

// NEWPM: SimplifyCFGPass
// NEWPM: SampleProfileLoaderPass

int func(int a) { return a; }
