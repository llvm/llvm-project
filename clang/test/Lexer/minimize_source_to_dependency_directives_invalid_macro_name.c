// RUN: %clang_cc1 -print-dependency-directives-minimized-source %s 2>&1 | FileCheck %s

#define 0 0
// CHECK: #define 0 0
