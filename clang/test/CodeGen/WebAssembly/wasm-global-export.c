// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

// Test export_name with explicit name on addrspace(1) global
int __attribute__((address_space(1))) exported_g
    __attribute__((export_name("global_g"))) = 42;

// Test export_name with explicit name on addrspace(0) memory global
int exported_mem __attribute__((export_name("mem_g"))) = 100;

// Test export_name on forward declaration propagating to definition
extern int var_propagate __attribute__((export_name("exported_propagate")));
int var_propagate = 102;

// CHECK: @exported_g = addrspace(1) global i32 42, align 4 #0
// CHECK: @exported_mem = global i32 100, align 4 #1
// CHECK: @var_propagate = global i32 102, align 4 #2

// CHECK: attributes #0 = { "wasm-export-name"="global_g" }
// CHECK: attributes #1 = { "wasm-export-name"="mem_g" }
// CHECK: attributes #2 = { "wasm-export-name"="exported_propagate" }
