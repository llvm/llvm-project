// RUN: %clang_cc1 -emit-llvm-only -verify %s
// expected-no-diagnostics

bool a() { return __is_pod(int); }
