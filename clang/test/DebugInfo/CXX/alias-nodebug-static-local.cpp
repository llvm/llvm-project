// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// Aliases of static locals without debug info must not crash or get an
// imported declaration.

// CHECK-DAG: @a = alias i32, ptr @_ZZ1fvE1s
// CHECK-DAG: @b = alias i32, ptr @_ZZ1gvE1s
// CHECK-NOT: DIImportedEntity

__attribute__((nodebug)) void f() { static int s; }
extern int a __attribute__((alias("_ZZ1fvE1s")));

void g() { __attribute__((nodebug)) static int s; }
extern int b __attribute__((alias("_ZZ1gvE1s")));
