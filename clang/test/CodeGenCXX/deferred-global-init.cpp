// RUN: %clang_cc1 -no-opaque-pointers %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s
// PR5967

extern void* foo;
static void* const a = foo;
void* bar() { return a; }

// CHECK: @_ZL1a = internal global i8* null

// CHECK-LABEL: define internal {{.*}}void @__cxx_global_var_init
// CHECK: load i8*, i8** @foo
// CHECK: ret void

// CHECK-LABEL: define internal {{.*}}void @_GLOBAL__sub_I_deferred_global_init.cpp
// CHECK: call {{.*}}void @__cxx_global_var_init()
// CHECK: ret void
