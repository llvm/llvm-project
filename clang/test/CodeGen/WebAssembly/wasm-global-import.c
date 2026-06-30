// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

// Test import_module and import_name
extern const int __attribute__((address_space(1))) imported_g
    __attribute__((import_module("js"), import_name("global_g")));

int get_import(void) { return imported_g; }

// CHECK: @imported_g = external addrspace(1) constant i32, align 4, !wasm.import.module ![[MD_MOD:[0-9]+]], !wasm.import.name ![[MD_NAME:[0-9]+]]
// CHECK: ![[MD_MOD]] = !{!"js"}
// CHECK: ![[MD_NAME]] = !{!"global_g"}
