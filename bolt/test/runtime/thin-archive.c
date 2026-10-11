// Test that BOLT can consume thin archives for runtime libraries.

// REQUIRES: system-linux,bolt-runtime

// RUN: rm -rf %t && mkdir -p %t/objects

// RUN: cd %t/objects && llvm-ar x %libbolt_rt_instr
// RUN: cd %t && llvm-ar rcT libbolt_rt_instr.a objects/*
// RUN: FileCheck --input-file %t/libbolt_rt_instr.a --check-prefix=THIN %s
// THIN: !<thin>

// RUN: %clang %cflags -no-pie -Wl,-q -o %t/exe %s
// RUN: llvm-bolt -o %t/exe.bolt %t/exe \
// RUN:     --instrument --instrumentation-file=%t/exe.fdata \
// RUN:     --runtime-instrumentation-lib=%t/libbolt_rt_instr.a
// RUN: %t/exe.bolt
// RUN: cat %t/exe.fdata | FileCheck %s
// CHECK: main 0 0 1

#include <stdio.h>

int main(int argc, char *argv[]) { puts("thin archive test"); }
