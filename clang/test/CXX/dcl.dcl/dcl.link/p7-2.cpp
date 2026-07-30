// RUN: %clang_cc1 -ast-print -o - %s | FileCheck %s

extern "C" int f(void);
// CHECK: extern "C" int f()

extern "C" int v;
// CHECK: extern "C" int v
