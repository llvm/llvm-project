// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -fexceptions -fcxx-exceptions -emit-llvm -o - -std=c++11 2>&1 | FileCheck %s

// Test code generation for Wasm's Emscripten (JavaScript-style) EH.

void noexcept_throw() noexcept {
  throw 3;
}

// CATCH-LABEL: define void @_Z14noexcept_throwv()
// CHECK:       %[[LPAD:.*]] = landingpad { ptr, i32 }
// CHECK-NEXT:                    catch ptr null
// CHECK-NEXT:  %[[EXN:.*]] = extractvalue { ptr, i32 } %[[LPAD]], 0
// CHECK-NEXT:  call void @__clang_call_terminate(ptr %[[EXN]])
