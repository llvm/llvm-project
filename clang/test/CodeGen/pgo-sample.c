// Test if PGO sample use passes are invoked.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s
// CHECK: Simplify the CFG
// CHECK: SROA
// CHECK: Combine redundant instructions
// CHECK: Remove unused exception handling info
// CHECK: Sample profile pass
