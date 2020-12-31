// RUN: %clang_cc1 -triple x86_64-linux -fsanitize=address -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ASAN
// RUN: %clang_cc1 -triple x86_64-linux -O2 -fsanitize=address -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ASAN
// RUN: %clang_cc1 -triple x86_64-linux -fsanitize=kernel-address -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,KASAN
// RUN: %clang_cc1 -triple x86_64-linux -O2 -fsanitize=kernel-address -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,KASAN
//
// Not all platforms support aliases - test for Linux only.

int global; // generate ctor for at least 1 global
int aliased_global; // KASAN ignored
extern int __attribute__((alias("aliased_global"))) __global_alias;

// Recursive alias:
int aliased_global_2; // KASAN ignored
extern int __attribute__((alias("aliased_global_2"))) global_alias_2;
extern int __attribute__((alias("global_alias_2"))) __global_alias_2_alias;

// Potential indirect alias:
struct input_device_id {
  unsigned long keybit[24];
  unsigned long driver_info;
};
struct input_device_id joydev_ids[] = { { {1}, 1234 } }; // KASAN ignored
extern struct input_device_id __attribute__((alias("joydev_ids"))) __mod_joydev_ids_device_table;

// ASAN: @aliased_global{{.*}} global { i32, [60 x i8] }{{.*}}, align 32
// ASAN: @aliased_global_2{{.*}} global { i32, [60 x i8] }{{.*}}, align 32
// ASAN: @joydev_ids{{.*}} global { {{.*}}[56 x i8] zeroinitializer }, align 32
// KASAN: @aliased_global{{.*}} global i32
// KASAN: @aliased_global_2{{.*}} global i32
// KASAN: @joydev_ids{{.*}} global [1 x {{.*}}i64 1234 }], align 16

// Check the aliases exist:
// CHECK: @__global_alias ={{.*}} alias
// CHECK: @global_alias_2 ={{.*}} alias
// CHECK: @__global_alias_2_alias ={{.*}} alias
// CHECK: @__mod_joydev_ids_device_table ={{.*}} alias

// CHECK-LABEL: define internal void @asan.module_ctor
// ASAN: call void @__asan_register_globals({{.*}}, i{{32|64}} 4)
// KASAN: call void @__asan_register_globals({{.*}}, i{{32|64}} 1)
// CHECK-NEXT: ret void

// CHECK-LABEL: define internal void @asan.module_dtor
// CHECK-NEXT: call void @__asan_unregister_globals
// CHECK-NEXT: ret void
