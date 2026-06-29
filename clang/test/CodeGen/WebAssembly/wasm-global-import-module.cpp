// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -verify -o - %s | FileCheck %s

extern "C" {
extern const int [[clang::address_space(1)]] imported_g
    __attribute__((annotate("wasm-import-module", "js")));
extern const int [[clang::address_space(1)]] bad_arity
    __attribute__((annotate("wasm-import-module"))); // expected-warning {{ignoring 'annotate("wasm-import-module", ...)' because it requires exactly one string literal argument}}
extern const int [[clang::address_space(1)]] bad_type
    __attribute__((annotate("wasm-import-module", 42))); // expected-warning {{ignoring 'annotate("wasm-import-module", ...)' because it requires exactly one string literal argument}}
extern const int [[clang::address_space(1)]] default_g;

int get() { return imported_g + bad_arity + bad_type + default_g; }
}

// CHECK: @imported_g = external addrspace(1) constant i32, align 4, !wasm.import.module ![[MD:[0-9]+]]
// CHECK: @bad_arity = external addrspace(1) constant i32, align 4
// CHECK: @bad_type = external addrspace(1) constant i32, align 4
// CHECK: @default_g = external addrspace(1) constant i32, align 4
// CHECK: ![[MD]] = !{!"js"}