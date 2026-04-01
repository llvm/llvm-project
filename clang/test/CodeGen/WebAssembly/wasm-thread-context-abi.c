// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

void use_stack() {
    int x;
    volatile int* ptr = &x;
}

void use_tls() {
    static __thread int x;
    volatile int* ptr = &x;
}