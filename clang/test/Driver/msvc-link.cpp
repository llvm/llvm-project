// REQUIRES: system-windows
//
// RUN: %clang -### %s -Ltest 2>&1 | FileCheck %s
//
// Test that user provided paths come before compiler-rt
// CHECK: "-libpath:test"
// CHECK: "-libpath:{{.*}}\\lib\\clang\\{{[0-9]+}}\\lib\\windows"
