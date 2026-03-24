// Test that P1689 format suppresses warnings during dependency scanning.
// This addresses https://github.com/llvm/llvm-project/issues/185394

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// Test P1689 format - should NOT emit warnings
// RUN: clang-scan-deps -format=p1689 -- %clang++ -std=c++20 -I%t -c %t/mylib.cppm -o %t/mylib.o 2>&1 | FileCheck %s

// CHECK-NOT: warning:
// CHECK: {
// CHECK: "revision"

//--- header.h
// Empty header for testing

//--- mylib.cppm
module;

export module mylib;

#include <header.h>