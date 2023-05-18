// REQUIRES: z3
// RUN: echo %clang_analyze_cc1 | FileCheck %s
// CHECK: -analyzer-constraints=z3
