// RUN: %clang_cc1 -triple x86_64-linux -fsanitize=address -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ASAN
// RUN: %clang_cc1 -triple x86_64-linux -fsanitize=kernel-address -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,KASAN
//
// Not all platforms support aliases - test for Linux only.

int global;                                                         // to generate ctor for at least 1 global
int aliased_global;                                                 // KASAN - ignore globals prefixed by aliases with __-prefix (below)
extern int __attribute__((alias("aliased_global"))) __global_alias; // KASAN - aliased_global ignored

// ASAN: @aliased_global{{.*}} global { i32, [60 x i8] }{{.*}}, align 32
// KASAN: @aliased_global{{.*}} global i32

// CHECK-LABEL: define internal void @asan.module_ctor
// ASAN: call void @__asan_register_globals({{.*}}, i{{32|64}} 2)
// KASAN: call void @__asan_register_globals({{.*}}, i{{32|64}} 1)
// CHECK-NEXT: ret void

// CHECK-LABEL: define internal void @asan.module_dtor
// CHECK-NEXT: call void @__asan_unregister_globals
// CHECK-NEXT: ret void
