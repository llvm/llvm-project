// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// PR5967

extern void* foo;
static void* const a = foo;
void* bar() { return a; }

// CHECK: @_ZL1a = internal global i8* null

// CHECK: define internal void @__cxx_global_var_init
// CHECK: load i8** @foo
// CHECK: ret void

// CHECK: define internal void @_GLOBAL__I_a
// CHECK: call void @__cxx_global_var_init()
// CHECK: ret void
