// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -target-feature +reference-types -emit-llvm -o - -std=c++11 | FileCheck %s
// RUN: %clang_cc1 %s -triple wasm64-unknown-unknown -target-feature +reference-types -emit-llvm -o - -std=c++11 | FileCheck %s

// Test that funcref can be used in C++ without crashing during codegen.
// See https://github.com/llvm/llvm-project/issues/176154
typedef void (*__funcref funcref_t)();

// Global funcref variables - test that codegen doesn't crash.
// CHECK-DAG: @fptr = global ptr addrspace(20) null
funcref_t fptr;

// CHECK-DAG: @fpt2 = global ptr addrspace(20) null
void (*__funcref fpt2)();

// CHECK-DAG: _Z2f1u11externref_t
void f1(__externref_t) {}

// CHECK-DAG: _Z2f2PU4AS20FvvE
void f2(funcref_t) {}
