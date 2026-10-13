// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

// Test import_module and import_name
extern const int __attribute__((address_space(1))) imported_g
    __attribute__((import_module("js"), import_name("global_g")));

int get_import(void) { return imported_g; }

// Test that defining a forward-declared imported global works and does not emit import attributes
extern const int __attribute__((address_space(1))) imported_def
    __attribute__((import_module("js"), import_name("global_def")));
const int __attribute__((address_space(1))) imported_def = 99;

// Test that defining a forward-declared imported function works and does not emit import attributes
extern void imported_fn(void)
    __attribute__((import_module("js"), import_name("fn")));
void imported_fn(void) {}

// CHECK: @imported_g = external addrspace(1) constant i32, align 4 #0
// CHECK: @imported_def = addrspace(1) constant i32 99, align 4{{$}}

// CHECK: define void @imported_fn()

// CHECK: attributes #0 = { "wasm-import-module"="js" "wasm-import-name"="global_g" }
// CHECK-NOT: "wasm-import-module"="js" "wasm-import-name"="global_def"
// CHECK-NOT: "wasm-import-module"="js" "wasm-import-name"="fn"



