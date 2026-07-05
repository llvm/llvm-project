// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fexceptions -fcxx-exceptions -target-feature +reference-types -target-feature +exception-handling -target-feature +multivalue -mllvm -wasm-enable-eh -exception-model=wasm -emit-llvm -o - %s | FileCheck %s

// Check if __builtin_wasm_throw and __builtin_wasm_rethrow are correctly
// invoked when placed in try-catch.

void throw_in_try(void *obj) {
  try {
    __builtin_wasm_throw(0, obj);
  } catch (...) {
  }
  // CHECK: invoke void @llvm.wasm.throw(i32 0, ptr %{{.*}})
}

void rethrow_in_try() {
  try {
  __builtin_wasm_rethrow();
  } catch (...) {
  }
  // CHECK: invoke void @llvm.wasm.rethrow()
}
