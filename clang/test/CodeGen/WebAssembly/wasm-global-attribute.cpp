// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

extern "C" {
int exported_g [[clang::wasm_global]] = 42;

extern const int imported_g [[clang::wasm_global]]
    __attribute__((import_module("env"), import_name("imported_g")));

int get_import(void) { return imported_g; }
}

// CHECK: @exported_g = addrspace(1) global i32 42, align 4
// CHECK: @imported_g = external addrspace(1) constant i32, align 4 #0
// CHECK: define{{.*}} @get_import()
// CHECK: load i32, ptr addrspace(1) @imported_g
// CHECK: attributes #0 = { "wasm-import-module"="env" "wasm-import-name"="imported_g" }