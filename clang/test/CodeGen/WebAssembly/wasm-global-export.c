// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

// Test export_name
int __attribute__((address_space(1))) exported_g
    __attribute__((export_name("global_g"))) = 42;

// CHECK: @exported_g = addrspace(1) global i32 42, align 4, !wasm.export.name ![[MD_EXPORT:[0-9]+]]
// CHECK: ![[MD_EXPORT]] = !{!"global_g"}
