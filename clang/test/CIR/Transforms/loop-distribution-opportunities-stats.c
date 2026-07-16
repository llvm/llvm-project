// REQUIRES: asserts
// RUN: %clang_cc1 -fclangir -emit-cir %S/loop-distribution-opportunities.c \
// RUN:   -o - | cir-opt -cir-loop-distribution -mlir-disable-threading \
// RUN:   -mlir-pass-statistics -o /dev/null 2>&1 | FileCheck %s

// CHECK: LoopDistribution
// CHECK: (S) 1 num-candidates - Number of loop distribution candidates
